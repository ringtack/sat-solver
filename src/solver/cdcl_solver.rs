use rand::Rng;
use std::{cmp::Ordering, mem, ops::BitAnd, time::Instant};

use fxhash::{FxHashMap, FxHashSet};
use log::{debug, info};
use mut_binary_heap::BinaryHeap;
use ordered_float::{OrderedFloat, Pow};
use slotmap::basic::{Iter, IterMut};

use crate::{
    dimacs::{self, sat_instance::SATInstance},
    solver::{
        types::lits_from_vars,
        util::{has_dup, vec_to_str, vec_with_size},
    },
};

use super::{
    assignment_trail::AssignmentStack,
    clause::{Clause, ClauseAllocator, ClauseKey, Reason},
    config::{ClauseDeletionConfig, DecisionConfig, OptConfig, RestartConfig, SolverConfig},
    stats::RuntimeStats,
    types::{DecisionLevel, LBool, Lit, SolveResult, SolveStatus, Var, F64, LBD, L_UNDEF, V_UNDEF},
    watch_list::{WatchList, Watcher},
};

pub struct CDCLSolver {
    ca: ClauseAllocator,
    /// Problem information: constraint clauses, learnt clauses.
    clauses: Vec<ClauseKey>,
    learnts: Vec<ClauseKey>,
    // (SEE NOTE BELOW) This formatting peculiarity can potentially affect the activity heap;
    // on construction, make sure not to add any variables that don't actually exist. To avoid
    // duplicate computation, this will store that resultant hash set.
    // real_vars: FxHashSet<Var>,
    /// NOTE: actually, we could just create a mapping from solver-specific variables to provided
    /// variables, in order to avoid this problem entirely.
    var_mapping: FxHashMap<Var, dimacs::sat_instance::Variable>,

    /// Search/inference fields.
    ///
    /// Current decision level in search.
    decision_level: DecisionLevel,
    /// Assignment stack during search and inference; will need to rewind on conflicts.
    trail: AssignmentStack,
    /// Watch list (i.e. occurrence list) for literals to track which clauses are watching them.
    watches: WatchList,

    // General options, decision heuristics, clause deletion, and restart policies
    conf: OptConfig,
    dh_conf: DecisionConfig,
    cd_conf: ClauseDeletionConfig,
    rs_conf: RestartConfig,

    /// Variable/Literal metadata.
    ///
    /// We use separate vectors, as opposed to one struct, to optimize cache accesses if we only
    /// need some subset of the data (which is almost always the case).
    ///
    /// For all following vectors (and temporary structures), if the key is either a Var or a Lit,
    /// it will be initialized to (nVars) or (nLits) size. Otherwise, it will be initially empty.
    /// TODO: figure out some initial capacity?
    ///
    /// Var -> assignment (if exists)
    assigned: Vec<LBool>,
    /// Var -> polarity (if exists)
    /// If phase save, store previously assigned polarities to use when deciding branch lits.
    polarity: Vec<bool>,
    /// VAR -> reason information
    /// Useful for conflict analysis (i.e. needed when iterating backwards to detect UIP)
    reasons: Vec<Reason>,
    /// Var -> activity
    /// Stores EVSIDs values, along with a mutable max heap to record highest order.
    acts: Vec<F64>,
    /// Stack to correspond with activities.
    act_heap: BinaryHeap<Var, F64>,

    /// Temporary computation structures, in order to prevent repetitive allocation/deallocation.
    ///
    /// Var -> bool
    /// Used to remember if a variable has already been seen in conflict analysis and clause
    /// minimization. Remember to clear using seen_to_clear!
    /// seen_to_clear!
    seen: Vec<bool>,
    /// Stack of vars
    /// Clear seen values after done.
    seen_to_clear: Vec<Var>,
    /// Stack of lits
    /// Used in conflict analysis for clause minimization
    analyze_stack: Vec<Lit>,
    /// Vec of lits
    /// Scratch space for reason lits (copy to/from here to avoid mutable ownership issues)
    reason_lits: Vec<Lit>,
    /// Vec of lits
    /// Scratch space for learnt lits
    learnt_lits: Vec<Lit>,

    /// Stats.
    stats: RuntimeStats,
    /// Elapsed time.
    start: Instant,
    /// Instance
    instance: String,

    // TODO: hacky, change later
    until_next_log: i64,
}

impl CDCLSolver {
    pub fn new(c: SolverConfig, instance: SATInstance) -> Self {
        // Convert instance into clauses, with the appropriate variable mapping
        let mut var_mapping = FxHashMap::default();
        let mut var = 0 as Var;
        let mut instance_vars = instance.vars.iter().collect::<Vec<_>>();
        instance_vars.sort();
        for i_var in instance_vars {
            var_mapping.insert(var, *i_var);
            var += 1;
        }
        // var is now the actual number of vars; use that to initialize all containers
        // Don't need +1, since vars (and thus lits too) start at 0.
        let n_vars = var as usize;
        let n_lits = lits_from_vars(n_vars);
        // Construct activity heap
        let acts = vec![OrderedFloat(1.); n_vars];
        let mut act_heap = BinaryHeap::with_capacity(n_vars);
        for v in 0..n_vars {
            act_heap.push(v as Var, OrderedFloat(1.));
        }

        let mut solver = Self {
            ca: ClauseAllocator::new(instance.n_clauses),
            clauses: vec![],
            learnts: vec![],
            var_mapping,
            decision_level: 0,
            trail: AssignmentStack::new(n_vars),
            watches: WatchList::new(n_lits),
            conf: c.opt_config(),
            dh_conf: c.decision_config(),
            cd_conf: c.clause_deletion_config(),
            rs_conf: c.restart_config(),
            assigned: vec![LBool::Undef; n_vars],
            polarity: vec![c.decision_config().prefer_true; n_vars],
            reasons: vec_with_size(n_vars, Reason::default()),
            acts,
            act_heap,
            seen: vec![false; n_vars],
            seen_to_clear: vec![],
            analyze_stack: vec![],
            reason_lits: vec![],
            learnt_lits: vec![],
            stats: RuntimeStats::default(),
            start: Instant::now(),
            instance: instance.instance.clone(),

            until_next_log: 0,
        };
        // Init solver with instance clauses
        solver.init(instance);
        solver
    }

    fn init(&mut self, instance: SATInstance) -> SolveStatus {
        // Reverse mapping necessary for converting instance variables to our mapping
        let mut rev_mapping = FxHashMap::default();
        for (var, i_var) in &self.var_mapping {
            rev_mapping.insert(*i_var, *var);
        }

        // Make clauses from instance
        for c in instance.clauses {
            // Convert lits into mapped lits
            let lits = c
                .lits
                .iter()
                .map(|i_lit| {
                    let var = *rev_mapping.get(&i_lit.abs()).unwrap();
                    Lit::new(var, *i_lit < 0)
                })
                .collect::<Vec<_>>();
            // If adding clause found empty clause, return error (since DL 0 still)
            if let SolveStatus::UNSAT = self.add_clause(lits) {
                return SolveStatus::UNSAT;
            }
        }

        SolveStatus::Unknown
    }

    pub fn solve(&mut self) -> SolveResult {
        loop {
            self.stats.starts += 1;
            let mut restart_lim = 0;
            if self.rs_conf.restart {
                restart_lim = self.luby();
            }
            debug!("Restarting execution with lim {restart_lim} (0 means infinite)");
            let res = self.search(restart_lim);
            if res != SolveStatus::Unknown {
                return SolveResult {
                    instance: self.instance.clone(),
                    status: res,
                    elapsed: self.start.elapsed(),
                    assignments: self.assignments(),
                };
            }
        }
    }

    pub fn search(&mut self, restart_lim: usize) -> SolveStatus {
        // Record stats
        let _n_conflicts = 0;

        loop {
            let conflict = self.propagate();

            if self.until_next_log == 0 {
                info!("[Elapsed: {:#?}] Current stats:", self.start.elapsed());
                info!(
                    "- Conflicts: {}, Propagations: {}, Decisions: {}, Rand Decisions: {}, Assignments: {} ({} total)",
                    self.stats.conflicts,
                    self.stats.propagations,
                    self.stats.decisions,
                    self.stats.rand_decisions,
                    self.n_assignments(),
                    self.n_vars(),
                );
                info!(
                    "- Constraints: {}, Learned clauses: {}, Learned literals: {}",
                    self.stats.n_clauses, self.stats.n_learnts, self.stats.n_learnt_lits,
                );
                info!(
                    "- Average watchers per literal: {:.5}, average literals per learnt clause: {}",
                    self.avg_watchers(),
                    self.stats
                        .n_learnt_lits
                        .checked_div_euclid(self.stats.n_learnts)
                        .unwrap_or_default()
                );

                self.until_next_log = 100_000;
            }
            // If no conflict, ultimately check if all variables are assigned, or esle
            // pick a condition to branch on.
            if let None = conflict {
                // If at base level (i.e. no decisions yet), try simplifying w/ learned
                // clauses; if we can't smplify further, formula is UNSAT
                if self.decision_level == 0 && !self.simplify() {
                    return SolveStatus::UNSAT;
                }

                // If at restart limit for conflicts, restart
                if restart_lim > 0 && self.stats.conflicts >= restart_lim as u64 {
                    // Backtrack to base level
                    self.backtrack(0);
                    return SolveStatus::Unknown;
                }

                // Decide new variable
                self.stats.decisions += 1;
                let next = self.decide();
                match next {
                    // If no next one found, we've found a satisfying assignment, so return
                    None => return SolveStatus::SAT,
                    Some(lit) => {
                        debug!("Deciding lit: {lit}");
                        self.make_decision(lit);
                    }
                }
            } else {
                let conflict_ck = conflict.unwrap();
                self.stats.conflicts += 1;
                self.stats.n_learnts += 1;

                // If conflict occurred at DL 0, oopsies unsat
                if self.decision_level == 0 {
                    return SolveStatus::UNSAT;
                }

                debug!("Trail: {:?}", vec_to_str(&self.trail.trail));

                debug!(
                    "(DL {}, lit {}) analyzing conflict with cause {:?}",
                    self.decision_level,
                    self.trail
                        .get(self.trail.dl_delim_idx(self.decision_level - 1)),
                    self.ca[conflict_ck],
                );
                // Analyze cause of conflict
                let (learnt_lits, bt_lvl) = self.analyze_conflicts(conflict_ck);
                // Backtrack to pre-conflict level
                self.backtrack(bt_lvl);

                // If just one clause, just add to trail as new implication (since it'll be implied
                // by BCP anyways)
                if learnt_lits.len() == 1 {
                    debug!(
                        "unit learned clause (lit {}), adding to trail",
                        learnt_lits[0],
                    );

                    self.add_to_trail(learnt_lits[0], None);
                } else {
                    debug!("Creating clause from {}", vec_to_str(&learnt_lits));

                    // Otherwise, create clause
                    let learnt_ck = self.create_clause(&learnt_lits, true);
                    let lbd = self.clause_lbd(&learnt_lits);
                    // Automatically drop learnt_c after bumping activity, since we won't need it
                    if {
                        let learnt_c = &mut self.ca[learnt_ck];
                        // Set LBD of learnt clause, and mark protected
                        learnt_c.lbd = lbd;
                        learnt_c.protected = true;

                        learnt_c.bump_activity(self.cd_conf.inc_var, self.cd_conf.rescale_lim)
                    } {
                        self.rescale_clause_activity();
                    }
                    // Create watchers for literal
                    self.learnts.push(learnt_ck);
                    self.attach_clause(learnt_ck);

                    debug!("Adding debug_asserting literal {} to trail", learnt_lits[0]);

                    // Bump clause activity, and add assigning literal to trail
                    self.add_to_trail(learnt_lits[0], Some(learnt_ck));
                }

                // If too many conflicts, delete some clauses based on sorting criteria:
                // first LBD, then activity, then size.
                // TODO: expand to others besides glucose
                if self.should_delete_clauses() {
                    debug!("Deleting clauses");
                    self.delete_clauses();
                }

                // Decay activities
                self.decay_clause_activity();
                self.decay_var_activity();
            }
        }
    }

    /// Statistics computations.
    ///
    /// Counts the number of assignments so far.
    pub fn n_assignments(&self) -> usize {
        self.assigned
            .iter()
            .map(|a| (*a != LBool::Undef) as usize)
            .reduce(|acc, b| acc + b)
            .unwrap()
    }

    /// Computes the average number of watchers per literal.
    pub fn avg_watchers(&self) -> f64 {
        let total = self
            .watches
            .occs
            .iter()
            .map(|v| v.len())
            .reduce(|acc, sz| acc + sz)
            .unwrap();
        total as f64 / self.watches.occs.len() as f64
    }

    // Emit assignments back in original DIMACS form.
    pub fn assignments(&self) -> Vec<Lit> {
        let mut og_assignments = Vec::with_capacity(self.n_vars());
        // For each assignment, map back to original value
        for (v, ass) in self.assigned.iter().enumerate() {
            og_assignments.push(Lit::new(
                *self.var_mapping.get(&(v as i64)).unwrap(),
                !bool::from(*ass),
            ));
        }
        og_assignments
    }

    // Implements BCP (unit propagation). This occurs as a result of a decision.
    fn propagate(&mut self) -> Option<ClauseKey> {
        // Record if conflict occurs
        let mut conflict = None;
        let mut n_props = 0;

        // Go until bcp_idx >= trail.size()
        while let Some(l) = self.get_next_bcp_lit() {
            n_props += 1;

            // Process all watchers for this literal
            conflict = self.propagate_process_watchers_for_lit(l);

            // If conflict occurred, set bcp idx to end of trail; this has the same effect of
            // breaking out of the loop (whose condition is bcp_idx < trail.len())
            if conflict.is_some() {
                self.trail.set_bcp_idx_to_trail_head();
            }
        }
        // Update stats
        self.stats.propagations += n_props;
        self.until_next_log -= n_props as i64;
        if self.until_next_log < 0 {
            self.until_next_log = 0
        }

        conflict
    }

    fn propagate_process_watchers_for_lit(&mut self, l: Lit) -> Option<ClauseKey> {
        let mut conflict = None;
        let mut watchers = self.watches.take_watchers(l);
        // Store counters to record current progress
        let (mut i, mut j) = (0, 0);
        let n_ws = watchers.len();

        debug!("Propagating {l} with {n_ws} watchers");

        'next_watcher: while i < n_ws {
            // better not have conflict if we runnin it back
            debug_assert!(conflict.is_none());

            let w_i = &mut watchers[i];
            debug!(
                "Watching clause: {:?} (blocker {})",
                &self.ca[w_i.ck], w_i.blocker
            );
            // If the clause is deleted, I oopsied and forgot to remove somewhere... so remove watcher
            // here
            if self.ca[w_i.ck].garbage {
                i += 1;
                continue;
            }

            // See if we can skip this clause; if blocker already assigned, we don't care
            if self.value(w_i.blocker) == LBool::True {
                debug!("Blocker {} true, skipping", w_i.blocker,);
                watchers[j] = watchers[i];
                i += 1;
                j += 1;
                continue;
            }

            // For invariant, make sure false lit is second value in clause; we want the first lit
            // always to be unassigned (and the false lit is assigned by definition from it being
            // on the trail)
            let neg_l = !l;
            let ck = w_i.ck;
            let (first, c_sz) = {
                let c = &mut self.ca[ck];

                // debug_assert!(c[0] != c[1]);
                if c[0] == neg_l {
                    debug!("Swapped {} with {}", c[0], c[1]);
                    c.lits.swap(0, 1);
                    // c[0] = c[1];
                    // c[1] = neg_l;
                }

                debug_assert!(c[1] == neg_l);
                i += 1;
                (c[0], c.size)
            };

            // If first watch is not blocker and is already true, then this clause is sat
            let w_first = Watcher::new(ck, first);
            if first != w_i.blocker && self.value(first) == LBool::True {
                debug!("First watcher {first} true, skipping");
                watchers[j] = w_first;
                j += 1;
                continue;
            }

            // Find new watcher, since either false/not yet assigned
            for k in 2..c_sz {
                // If not false, update that watcher's list too, and continue
                let v = self.value(self.ca[ck][k]);
                if v != LBool::False {
                    let c = &mut self.ca[ck];
                    c[1] = c[k];
                    c[k] = neg_l;
                    self.watches.get_watchers(!c[1]).push(w_first);

                    debug!(
                        "Added watcher (blocker {}) to lit {} (val: {})",
                        w_first.blocker, !c[1], v
                    );

                    continue 'next_watcher;
                }
            }

            watchers[j] = w_first;
            j += 1;
            // If we didn't find a new watcher, it's a unit clause: check if either
            // conflict, or a new implication
            if self.value(first) == LBool::False {
                debug!("Got conflict for {first}");

                conflict = Some(ck);
                // Copy remaining watches over, in case we've skipped some watches when we find new
                // watchers; this also has the effect of breaking out of the loop, since i == n_ws
                while i < n_ws {
                    watchers[j] = watchers[i];
                    i += 1;
                    j += 1;
                }
            } else {
                debug!("Adding {first} to trail with cause {:?}", self.ca[ck]);
                // If no conflict, we got a new implication, so add to trail and assign it
                self.add_to_trail(first, Some(ck));
            }
        }

        debug!("Done propagating {l}");

        // Truncate watcher's list to remove any we've skipped to find new watchers
        watchers.truncate(j);
        self.watches.set_watchers(l, watchers);

        // Report if conflict occurred
        conflict
    }

    // Simplifies the clause database by removing satisfied clauses.
    // Returns whether the format is either unknown (true), or UNSAT (false).
    fn simplify(&mut self) -> bool {
        debug!("Attempting simplify");

        // if propagating -> conflict, then conflicting units -> UNSAT
        if let Some(_) = self.propagate() {
            return false;
        }

        // Remove satisfied clauses
        // TODO: verify how well this works
        self.remove_satisfied_learnts();
        if self.conf.remove_satisfied {
            self.remove_satisfied_constraints();
        }
        self.rebuild_heap();
        true
    }

    // Decide the next branch following activity heuristic.
    fn decide(&mut self) -> Option<Lit> {
        let mut _act = OrderedFloat(0.);

        let mut next = V_UNDEF;
        if self.dh_conf.rand_var && rand::thread_rng().gen::<f64>() < self.dh_conf.rand_f {
            self.stats.rand_decisions += 1;
            next = rand::thread_rng().gen_range(0..(self.n_vars() as i64));
            debug!("rand decision");
        }

        // Iterate until either heap empty, or non-assigned next var found
        while next == V_UNDEF || self.assigned[next as usize] != LBool::Undef {
            // if empty, mark next as undefined again and leave
            if self.act_heap.is_empty() {
                next = V_UNDEF;
                break;
            }
            (next, _act) = self.act_heap.pop_with_key().unwrap();
        }
        // If undef, denote as such
        if next == V_UNDEF {
            None
        } else {
            // Check that act was actually max act
            // debug_assert!(
            //     self.acts
            //         .iter()
            //         .enumerate()
            //         .map(|(v, act)| {
            //             if self.assigned[v] != LBool::Undef {
            //                 _act >= *act
            //             } else {
            //                 true
            //             }
            //         })
            //         .all(|b| b),
            //     "did not get max act (got {} for {}, expected {})",
            //     _act,
            //     next,
            //     self.acts.iter().max().unwrap()
            // );

            let mut polarity = false;
            if self.dh_conf.rand_pol && rand::thread_rng().gen::<f64>() < self.dh_conf.rand_f {
                polarity = rand::thread_rng().gen_bool(0.5);
            } else {
                // Get polarity from previous decisions
                polarity = self.polarity[next as usize];
            }
            Some(Lit::new(next, polarity))
        }
    }

    // Analyze conflicts if one occurs. Returns the literals of the new clause, and the decision
    // level to which to backtrack.
    fn analyze_conflicts(&mut self, ck: ClauseKey) -> (Vec<Lit>, DecisionLevel) {
        // Record the current debug_asserting literal.
        let mut a_lit = None;
        let mut vars_to_bump = Vec::with_capacity(32);
        self.learnt_lits.clear();
        self.learnt_lits.resize(1, Lit::default());
        self.seen_to_clear.clear();

        for s in &self.seen {
            debug_assert!(*s == false);
        }

        // Iterate backwards through trail_idx until we've reverse-BFS'ed to the first UIP.
        let mut trail_idx = self.trail_size() - 1;
        let mut trail_ctr = 0;
        let mut curr_ck = Some(ck);
        let mut rescale_clauses = false;
        loop {
            // curr ck better be some reason, otherwise we fucked up somewhere
            debug_assert!(curr_ck.is_some());

            // Compute reason
            let learnt = {
                let c = &self.ca[curr_ck.unwrap()];

                debug!("Reason (ctr: {trail_ctr}): {:?}", c);
                debug_assert!(!has_dup(&c.lits));

                self.reason_lits.resize(c.lits.len(), Lit::default());
                self.reason_lits.copy_from_slice(&c.lits);
                c.learnt
            };
            // If learnt, bump activity to prioritize in clause deletion
            if learnt {
                // Record whether we need to re-scale clauses
                rescale_clauses = {
                    // Limit scope here to prevent borrow checker from angies
                    let c = &mut self.ca[curr_ck.unwrap()];
                    c.bump_activity(self.cd_conf.inc_var, self.cd_conf.rescale_lim)
                };
            }

            // Check every clause literal except the first (unless a_lit is None, i.e. first cause)
            // We maintain invariant that the first literal in the clause is the debug_asserting literal,
            // so we can skip. If we don't, since we've already marked not seen, this causes an
            // infinite loop.
            let s_idx = if a_lit.is_none() { 0 } else { 1 };
            for lit in self.reason_lits[s_idx..].into_iter() {
                let var = lit.var();
                let lvl = self.level(var);

                debug!(
                    "Should visit var {var} (lvl {lvl}, pos {:?})? ({}, {}). Learn? {}",
                    self.trail.trail.iter().position(|l| l.var() == lit.var()),
                    lvl > 0,
                    !self.seen[lit.var_idx()],
                    lvl < self.decision_level
                );

                // Skip 0-DL lits, since those would've already caused UNSATs
                // && self.assigned[lit.var_idx()] != LBool::Undef
                if lvl > 0 && !self.seen[lit.var_idx()] {
                    // If not yet seen, bump activity
                    self.seen[lit.var_idx()] = true;
                    vars_to_bump.push(var);
                    // If at/above DL, increase trail counter to clear
                    if lvl >= self.decision_level {
                        trail_ctr += 1;
                    } else {
                        // Otherwise, add to learned literals
                        self.learnt_lits.push(*lit);
                    }
                }
            }

            // trail_idx = self.trail_size() - 1;
            // After processing each literal within the reason clause, find next clause to parse
            let tmp = self.trail.get(trail_idx).var_idx();
            debug!(
                "{tmp} seen? {} (trail: {}, trail_idx: {})",
                self.seen[tmp],
                vec_to_str(&self.trail.trail),
                trail_idx
            );
            while !self.seen[self.trail.get(trail_idx).var_idx()] {
                debug!("{} not seen, going back", self.trail.get(trail_idx).var());
                trail_idx -= 1;
            }
            let l = self.trail_at(trail_idx);
            // Update debug_asserting lit
            a_lit = Some(l);
            // Get new reason
            curr_ck = self.reason_ref(l.var()).ck;
            // Clear seen-ness from this boi
            self.seen[l.var_idx()] = false;
            trail_ctr -= 1;

            debug!("New debug_asserting lit {l} with ctr {trail_ctr}");

            // If curr_ck is none, the trail counter must also be done, so we break (i.e. this is
            // the UIP)
            debug_assert!(
                curr_ck.is_some() || trail_ctr == 0,
                "curr_ck: {:?}, trail_ctr: {trail_ctr}",
                curr_ck
            );
            // Check if reached UIP
            if trail_ctr == 0 {
                debug!("Found UIP (a_lit: {}), done backtracking", a_lit.unwrap());
                break;
            }
        }

        if rescale_clauses {
            self.rescale_clause_activity();
        }
        for var in &vars_to_bump {
            self.bump_var_activity(*var);
        }

        // Set first learnt lit to negation of debug_asserting lit (since we want to prevent this
        // implication in the future).)
        self.learnt_lits[0] = !a_lit.unwrap();

        // Mark seen vars to be cleared
        for l in &self.learnt_lits {
            self.seen_to_clear.push(l.var());
        }

        let mut v = mem::take(&mut self.learnt_lits);
        // Minimize clause
        self.conflict_clause_minimization(&mut v);
        let _ = mem::replace(&mut self.learnt_lits, v);

        // Find backtrack level: if unit learnt lit, backtrack back to base DL to propagate
        let bt_lvl = if self.learnt_lits.len() == 1 {
            0
        } else {
            // Otherwise, find highest DL in learnt lits (that's not the debug_asserting lit)
            let (max_i, max_lvl) = self.learnt_lits[1..]
                .into_iter()
                .enumerate()
                .max_by(|(_, l1), (_, l2)| self.level(l1.var()).cmp(&self.level(l2.var())))
                // +1 since max_i is enumerated from 1.., so it's one too low
                .map(|(i, l)| (i + 1, self.level(l.var())))
                .unwrap();
            debug!("max_i: {max_i}");
            //  Swap 2nd learned lit with the max level lit;
            self.learnt_lits.swap(1, max_i);
            debug_assert!(max_lvl == self.level(self.learnt_lits[1].var()));
            debug_assert!(!has_dup(&self.learnt_lits));
            max_lvl
        };

        // Clear seen variables
        for v_to_clear in &self.seen_to_clear {
            self.seen[*v_to_clear as usize] = false;
        }
        self.seen_to_clear.clear();

        (self.learnt_lits.clone(), bt_lvl)
    }

    /// Minimizes a clause by filtering redundant literals.
    fn conflict_clause_minimization(&mut self, lits: &mut Vec<Lit>) {
        let abs_lvl = self.abstract_levels(&lits);
        // Only include if decision variable, or not redundant:
        lits.retain(|l| self.reason(l.var()).ck.is_none() || !self.is_redundant(*l, abs_lvl));
    }

    // Backtrack to the desired decision level, cleaning up the trail as necessary.
    fn backtrack(&mut self, dl: DecisionLevel) {
        if self.decision_level <= dl {
            return;
        }

        // Get level delimiter trail index for this level
        let lvl_dlm_idx = self.trail.dl_delim_idx(dl);

        debug!(
            "Backtracking to DL {dl} (idx: {}, current trail: {})",
            lvl_dlm_idx,
            vec_to_str(&self.trail.trail)
        );

        while self.trail_size() > lvl_dlm_idx {
            let lit = self.pop_trail();
            let var = lit.var_idx();

            debug!(
                "Popped {lit} (lvl {}), unassigning back to LUndef",
                self.level(lit.var())
            );

            // Unassign this variable
            self.assigned[var] = LBool::Undef;
            // Clear its reason; it'll probably have a different one next time
            self.reasons[var] = Reason::default();
            // If we're phase saving, record the polarity
            if self.conf.save_phases {
                self.polarity[var] = lit.sign();
            }
            // Since we've picked this before, we removed from the activity max heap. Re-add.
            self.act_heap.push(lit.var(), self.acts[var]);
        }

        // After clearing the trail, update bcp to trail head
        self.trail.set_bcp_idx_to_trail_head();
        // Shrink tail down to the level specified (so if lvl = 2, truncate down to lvl+1)
        self.trail.dl_delim_idxs.truncate(dl as usize);
        self.decision_level = dl;

        debug!("Set DL back to {dl}");
    }

    /// Clause functions
    ///
    /// Adds a constraint clause.
    fn add_clause(&mut self, lits: Vec<Lit>) -> SolveStatus {
        let mut lits = lits;
        lits.sort();

        let mut j = 0;
        let mut prev_lit = L_UNDEF;
        for i in 0..lits.len() {
            let lit = lits[i];
            let v = self.value(lit);
            // If already decided, or (p ^ !p) in same clause, don't add clause
            if v == LBool::True || lit == !prev_lit {
                return SolveStatus::Unknown;
            }
            // Otherwise, only keep if not already decided false and lit not same as prev
            if v != LBool::False && lit != prev_lit {
                prev_lit = lit;
                lits[j] = prev_lit;
                j += 1;
            }
        }
        lits.truncate(j);

        // If no lits left, this is UNSAT
        match lits.len() {
            0 => return SolveStatus::UNSAT,
            // No reason here, since was decided on addition (i.e. dl == 0)
            1 => {
                self.add_to_trail(lits[0], None);
                if let Some(_) = self.propagate() {
                    return SolveStatus::UNSAT;
                }
            }
            _ => {
                // Add to watchlist and slotmap
                let ck = self.create_clause(&lits, false);
                self.clauses.push(ck);
                self.attach_clause(ck);

                self.stats.n_clauses += 1;
            }
        };

        SolveStatus::Unknown
    }

    /// Attaches a clause to watchlists.
    fn attach_clause(&mut self, ck: ClauseKey) {
        let (c0, c1, c_sz, learnt) = {
            let c = &self.ca[ck];
            (c[0], c[1], c.size, c.learnt)
        };
        debug_assert!(c_sz > 1);
        self.watches.add_watcher(!c0, Watcher::new(ck, c1));
        self.watches.add_watcher(!c1, Watcher::new(ck, c0));
        if learnt {
            self.stats.n_learnt_lits += c_sz as u64;
        } else {
            self.stats.n_clause_lits += c_sz as u64;
        }
    }

    /// Remove satisfied clauses from the constraint database.
    /// TODO: turn on/off and see if it affects correctness
    fn remove_satisfied_constraints(&mut self) {
        let mut j = 0;
        for i in 0..self.clauses.len() {
            let ck = self.clauses[i];
            let c = &self.ca[ck];
            // If satisfied, remove clause
            if self.satisfied(c) {
                self.remove_clause(ck, c[0], c[1]);
            } else {
                self.clauses[j] = ck;
                j += 1;
            }
        }
        self.clauses.truncate(j);
    }

    /// Remove satisfied clauses from the learnt clauses database.
    fn remove_satisfied_learnts(&mut self) {
        let mut j = 0;
        for i in 0..self.learnts.len() {
            let ck = self.learnts[i];
            let c = &self.ca[ck];
            // If satisfied, remove clause
            if self.satisfied(c) {
                self.remove_clause(ck, c[0], c[1]);
            } else {
                self.learnts[j] = ck;
                j += 1;
            }
        }
        self.learnts.truncate(j);
    }

    /// Remove a clause (i.e. its watchers) from the solver.
    fn remove_clause(&mut self, ck: ClauseKey, c0: Lit, c1: Lit) {
        debug!("Removing clause {:?}", self.ca[ck]);

        // Remove watchers for c0 and c1
        let w0 = Watcher::new(ck, c1);
        let w1 = Watcher::new(ck, c0);
        self.watches.remove_watcher(!c0, w0);
        self.watches.remove_watcher(!c1, w1);
        // TODO: see if I need to remove from reasons? and if I need to free?
    }

    /// Check if a clause is satisfied.
    fn satisfied(&self, c: &Clause) -> bool {
        c.lits.iter().any(|l| self.value(*l) == LBool::True)
    }

    /// Check if we should remove clauses.
    fn should_delete_clauses(&self) -> bool {
        // TODO: expand beyond glucose
        self.stats.conflicts % (self.cd_conf.u + self.stats.deletions * self.cd_conf.k) == 0
    }

    /// Deletes clauses:
    /// - Preserve all binary and just-added learnt clauses
    /// - Sort by lowest LBD first
    /// - Sort by highest activity
    fn delete_clauses(&mut self) {
        let mut learnts = self.learnts.clone();

        debug!("Start clauses: {}", learnts.len());

        // Sort first by LBD, then highest activity, then by size. First elements are the best.
        learnts.sort_by(|c1, c2| {
            let c1 = &self.ca[*c1];
            let c2 = &self.ca[*c2];
            if c1.lbd > c2.lbd {
                Ordering::Greater
            } else if c2.lbd > c1.lbd {
                Ordering::Less
            } else if c1.act < c2.act {
                Ordering::Greater
            } else if c2.act < c1.act {
                Ordering::Less
            } else if c1.lits.len() > c2.lits.len() {
                Ordering::Greater
            } else if c2.lits.len() > c1.lits.len() {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        });
        debug!(
            "Learnt vals: {}",
            vec_to_str(
                &learnts
                    .iter()
                    .map(|ck| self.ca[*ck].lbd)
                    .collect::<Vec<_>>()
            )
        );

        // Compute number to retain
        let n_to_retain = (learnts.len() as f64 * self.cd_conf.keep_f) as usize;

        let mut retained = 0;
        // Clauses to remove
        let mut ck_to_remove = Vec::with_capacity(n_to_retain);
        learnts.iter().for_each(|ck| {
            let c = &mut self.ca[*ck];
            // If c is protected or binary, don't do anything (but mark as no longer protected)
            if c.protected || c.lits.len() == 2 {
                c.protected = false;
                retained += 1;
                return;
            }
            // Otherwise, if surpassed n_to_retain, mark for removal
            if retained >= n_to_retain {
                c.garbage = true;
                // Add to be removed
                ck_to_remove.push(ck);
            } else {
                // Otherwise, increment retained
                retained += 1;
            }
        });

        // Remove clauses
        let mut deleted_clauses = 0;
        let mut deleted_lits = 0;
        self.learnts.retain(|ck| {
            let c = &self.ca[*ck];
            if c.garbage {
                deleted_clauses += 1;
                deleted_lits += c.lits.len() as u64;
                false
            } else {
                debug!("lbd: {} (DL {})", c.lbd, self.decision_level);
                true
            }
        });

        debug!("End clauses after deletion: {}", self.learnts.len());

        // Update stats
        self.stats.deletions += 1;
        self.stats.n_learnts -= deleted_clauses;
        self.stats.n_learnt_lits -= deleted_lits;
    }

    /// Auxiliary structure methods
    ///
    /// Add to the trail, updating the associated reason, dl, and assignment
    fn add_to_trail(&mut self, lit: Lit, ck: Option<ClauseKey>) {
        // Make sure not already assigned
        debug_assert!(
            self.value(lit) == LBool::Undef,
            "lit {lit} assigned to {} already",
            self.value(lit)
        );

        // If +Lit (i.e. sign == false), assign true; otherwise, assign false
        self.assigned[lit.var_idx()] = LBool::from_sign(!lit.sign());

        debug!(
            "Assigning {} to {} (cause: {:?}, lvl: {})",
            lit.var(),
            self.assigned[lit.var_idx()],
            ck,
            self.decision_level,
        );

        // Add reason to antecedent graph, and push to trail
        self.reasons[lit.var_idx()] = Reason {
            ck,
            dl: self.decision_level,
        };
        self.trail.push(lit);
    }

    /// Makes the decision for lit. Does all of add_to_trail, while updating the level
    /// delimiters.
    fn make_decision(&mut self, lit: Lit) {
        self.decision_level += 1;
        self.trail.dl_delim_idxs.push(self.trail_size());
        self.add_to_trail(lit, None);

        debug_assert!(self.decision_level as usize == self.trail.dl_delim_idxs.len());
    }

    /// Rebuild the heap from the activity list.
    fn rebuild_heap(&mut self) {
        let n_vars = self.n_vars();
        self.act_heap = BinaryHeap::with_capacity(n_vars);
        for var in 0..n_vars {
            // Only add if currenly unassigned
            if self.assigned[var] == LBool::Undef {
                self.act_heap.push(var as Var, self.acts[var]);
            }
        }
    }

    /// Remove all watchers for a lit that appear in a set .
    fn remove_from_watchers(&mut self, l: Lit, set: &FxHashSet<Watcher>) {
        let watchers = self.watches.get_watchers(l);
        debug!("watchers before: {}", watchers.len());
        watchers.retain(|w| !set.contains(w));
        debug!("watchers after: {}", watchers.len());
    }

    /// Literal/Variable accesses
    ///
    /// Calculate value given a literal
    fn value(&self, l: Lit) -> LBool {
        self.assigned[l.var_idx()] ^ LBool::from(l.sign() as u8)
    }

    /// Get DL for this variable's assignment.
    fn level(&self, v: Var) -> DecisionLevel {
        self.reason(v).dl
    }

    /// Get clause reason for this variable's assignment.
    fn reason_clause(&self, v: Var) -> Option<ClauseKey> {
        self.reason(v).ck
    }

    /// Get reason for this var
    fn reason(&self, v: Var) -> Reason {
        self.reasons[v as usize]
    }

    /// Get ref to reason for this var
    fn reason_ref(&self, v: Var) -> &Reason {
        &self.reasons[v as usize]
    }

    /// Get mut ref to reason for this var
    fn _reason_ref_mut(&mut self, v: Var) -> &mut Reason {
        &mut self.reasons[v as usize]
    }

    /// Computes the total number of variables.
    fn n_vars(&self) -> usize {
        self.var_mapping.len()
    }

    /// Computes the abstract level for a single literal.
    fn abstract_level(&self, v: Var) -> usize {
        1 << (self.level(v).bitand(31))
    }

    /// Computes the abstract level for a collection of literals, where abstract level
    /// simply represents the denoted levels in bit form.
    fn abstract_levels(&self, lits: &[Lit]) -> usize {
        let mut abs_lvl = 0;
        lits.iter()
            .for_each(|l| abs_lvl |= self.abstract_level(l.var()));
        abs_lvl
    }

    /// Computes the LBD of a collection of literals.
    fn clause_lbd(&self, lits: &[Lit]) -> LBD {
        let mut uniq = FxHashSet::default();
        for l in lits {
            uniq.insert(self.level(l.var()));
        }

        debug!("{} (DL {})", uniq.len() as LBD, self.decision_level);
        debug!(
            "{}",
            vec_to_str(&lits.iter().map(|l| self.level(l.var())).collect::<Vec<_>>())
        );

        uniq.len() as LBD
    }

    /// Determines whether a literal is redundant.
    fn is_redundant(&mut self, l: Lit, abs_lvls: usize) -> bool {
        self.analyze_stack.clear();
        self.analyze_stack.push(l);
        let top = self.seen_to_clear.len();
        while self.analyze_stack.len() > 0 {
            let l = self.analyze_stack.pop().unwrap();
            assert!(self.reason_clause(l.var()).is_some());

            let ck = self.reason_clause(l.var()).unwrap();
            let c = &self.ca[ck];
            // For every literal in the clause, check if it's redundant
            for c_lit in &c.lits[1..] {
                let var = c_lit.var();
                let lvl = self.level(var);
                if !self.seen[c_lit.var_idx()] && lvl > 0 {
                    // If it had some reason, and it's on one of the decision levels we're interested
                    // in, then we don't need to process it (since we already have variables to
                    // learn from)
                    if self.reason_clause(var).is_some()
                        && (self.abstract_level(var).bitand(abs_lvls) != 0)
                    {
                        self.seen[c_lit.var_idx()] = true;
                        self.analyze_stack.push(*c_lit);
                        self.seen_to_clear.push(var);
                    } else {
                        // Otherwise, it provides useful information, so keep
                        for v in &self.seen_to_clear[top..] {
                            self.seen[*v as usize] = false;
                        }
                        self.seen_to_clear.truncate(top);
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Luby restart computation.
    fn luby(&self) -> usize {
        // Find subsequence for current number of restarts
        let mut n_restarts = self.stats.starts;
        let (mut sz, mut seq) = (1, 0);
        while sz < n_restarts + 1 {
            seq += 1;
            sz = 2 * sz + 1;
        }

        while sz - 1 != n_restarts {
            sz = (sz - 1) >> 1;
            seq -= 1;
            n_restarts = n_restarts % sz;
        }

        debug!(
            "Luby: {}",
            self.rs_conf.u * (self.rs_conf.scale.pow(seq) as usize)
        );
        self.rs_conf.u * (self.rs_conf.scale.pow(seq) as usize)
    }

    /// Statistics computations for activity, LBD, etc.
    ///
    /// Rescales clause activity.
    fn rescale_clause_activity(&mut self) {
        let rescale_f = self.cd_conf.rescale_f;
        for (_, c) in self.clause_iter_mut() {
            c.act *= rescale_f;
        }
        self.cd_conf.inc_var *= self.cd_conf.rescale_f;
    }

    /// Bumps the activity of the variable. Returns whether activity scaling is needed.
    fn bump_var_activity(&mut self, v: Var) {
        let var = v as usize;
        self.acts[var] *= self.dh_conf.inc_var;
        // If exceeds limit, rescale all activities and rebuild heap.
        let lim = OrderedFloat(self.dh_conf.rescale_lim);
        if self.acts[var] >= lim {
            debug!(
                "Got act {} (inc var {}), rescaling variables by {}",
                self.acts[var], self.dh_conf.inc_var, self.dh_conf.rescale_f
            );
            for a in &mut self.acts {
                // If NaN/Inf, manually reset :/ makes EVSIDS kinda unreliable ugh
                if !a.is_finite() {
                    *a = OrderedFloat(1.);
                } else {
                    *a *= self.dh_conf.rescale_f;
                }
            }
            // If not finite, reset inc var
            if !self.dh_conf.inc_var.is_finite() {
                self.dh_conf.inc_var = 1.;
            } else {
                self.dh_conf.inc_var *= self.dh_conf.rescale_f;
            }

            debug!(
                "new acc: {}, new inc_var: {}",
                self.acts[var], self.dh_conf.inc_var
            );

            self.rebuild_heap();
        }

        // If in heap, update (it might not be in heap if we're updating during backtrack)
        if let Some(mut act) = self.act_heap.get_mut(&v) {
            *act = self.acts[var];
        }
    }

    /// Decays the clause activity scale factor (i.e. makes others less active comparatively).
    fn decay_clause_activity(&mut self) {
        self.cd_conf.inc_var *= self.cd_conf.f;
    }

    /// Decays the variable activity scale factor (i.e. makes others less active comparatively).
    fn decay_var_activity(&mut self) {
        self.dh_conf.inc_var *= self.dh_conf.f;
    }

    /// Simple wrappers around field methods.
    ///
    /// Creates a clause.
    fn create_clause(&mut self, lits: &[Lit], learnt: bool) -> ClauseKey {
        self.ca.create_clause(lits, learnt)
    }

    fn _clause_iter(&mut self) -> Iter<'_, ClauseKey, Clause> {
        self.ca.iter()
    }

    fn clause_iter_mut(&mut self) -> IterMut<'_, ClauseKey, Clause> {
        self.ca.iter_mut()
    }

    /// All propagated iff bcp_idx >= trail.size()
    fn _all_propagated(&self) -> bool {
        self.trail.bcp_idx_at_end()
    }

    /// Gets the next bcp lit and increases the bcp_idx, if it exists
    fn get_next_bcp_lit(&mut self) -> Option<Lit> {
        self.trail.get_next_bcp_lit()
    }

    // Gets the trail size.
    fn trail_size(&self) -> usize {
        self.trail.trail.len()
    }

    // Pops from the trail.
    fn pop_trail(&mut self) -> Lit {
        self.trail.trail.pop().unwrap()
    }

    // Indexes into the trail.
    fn trail_at(&self, idx: usize) -> Lit {
        self.trail.get(idx)
    }
}
