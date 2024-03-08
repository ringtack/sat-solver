use std::{borrow::BorrowMut, cell::RefCell, env::vars, mem};

use fxhash::FxHashMap;
use log::debug;
use mut_binary_heap::BinaryHeap;
use slotmap::basic::{Iter, IterMut};

use crate::{
    dimacs::{
        self,
        sat_instance::{Literal, SATInstance},
    },
    solver::{types::lits_from_vars, util::vec_with_size},
};

use super::{
    assignment_trail::AssignmentStack,
    clause::{Clause, ClauseAllocator, ClauseKey, Reason},
    config::{ClauseDeletionConfig, DecisionConfig, OptConfig, RestartConfig, SolverConfig},
    stats::RuntimeStats,
    types::{DecisionLevel, LBool, Lit, SolveStatus, Var, F64, L_UNDEF, V_UNDEF},
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
    /// Lit -> reason information
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
        let acts = vec![F64::new(0.).unwrap(); n_vars];
        let mut act_heap = BinaryHeap::with_capacity(n_vars);
        for v in 0..n_vars {
            act_heap.push(v as Var, F64::new(0.).unwrap());
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
            polarity: vec![false; n_vars],
            reasons: vec_with_size(n_lits, Reason::default()),
            acts,
            act_heap,
            seen: vec![false; n_vars],
            seen_to_clear: vec![],
            analyze_stack: vec![],
            reason_lits: vec![],
            learnt_lits: vec![],
            stats: RuntimeStats::default(),
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

    pub fn solve(&mut self) -> SolveStatus {
        // Record stats
        let n_conflicts = 0;
        self.stats.starts += 1;

        // TODO: integrate restarts into this
        loop {
            let conflict = self.propagate();
            // If no conflict, ultimately check if all variables are assigned, or esle
            // pick a condition to branch on.
            if let None = conflict {
                // TODO: check if # conflicts has reached some point

                // If at base level (i.e. no decisions yet), try simplifying w/ learned
                // clauses
                if self.decision_level == 0 {
                    // If we can't simplify further, formula is UNSAT
                    if !self.simplify() {
                        return SolveStatus::UNSAT;
                    }
                }

                // Decide new variable
                self.stats.decisions += 1;
                let next = self.decide();
                debug!("Deciding lit: {:?}", next);
                match next {
                    // If no next one found, we've found a satisfying assignment, so return
                    None => return SolveStatus::SAT,
                    Some(lit) => {
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

                // Analyze cause of conflict
                let (learnt_lits, bt_lvl) = self.analyze_conflicts(conflict_ck);
                // Backtrack to pre-conflict level
                self.backtrack(bt_lvl);

                // If just one clause, just add to trail as new implication (since it'll be implied
                // by BCP anyways)
                if learnt_lits.len() == 1 {
                    self.add_to_trail(learnt_lits[0], None);
                } else {
                    // Otherwise, create clause
                    let learnt_ck = self.create_clause(&learnt_lits, true);
                    // Automatically drop learnt_c after bumping activity, since we won't need it
                    if {
                        let learnt_c = &mut self.ca[learnt_ck];
                        learnt_c.bump_activity(self.cd_conf.inc_var, self.cd_conf.rescale_lim)
                    } {
                        self.rescale_clause_activity();
                    }
                    // Bump clause activity, and add assigning literal to trail
                    self.add_to_trail(learnt_lits[0], Some(learnt_ck));
                    // Create watchers for literal
                    self.learnts.push(learnt_ck);
                    self.attach_clause(learnt_ck);
                }

                // Decay activities
                self.decay_clause_activity();
                self.decay_var_activity();
            }
        }
    }

    // Emit assignments back in original DIMACS form.
    pub fn assignments(&self) -> Vec<Lit> {
        let mut og_assignments = Vec::with_capacity(self.n_vars());
        // For each assignment, map back to original value
        for (v, ass) in self.assigned.iter().enumerate() {
            og_assignments.push(Lit::new(
                *self.var_mapping.get(&(v as i64)).unwrap(),
                bool::from(*ass),
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

            debug!("Propagating {}", l.to_string());
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
        // TODO: update when to simplify i.e. clause delete here?

        conflict
    }

    fn propagate_process_watchers_for_lit(&mut self, l: Lit) -> Option<ClauseKey> {
        let mut conflict = None;
        let mut watchers = self.watches.take_watchers(l);
        // Store counters to record current progress
        let (mut i, mut j) = (0, 0);
        let n_ws = watchers.len();
        'next_watcher: while i < n_ws {
            let w_i = &mut watchers[i];
            debug!("Watching clause: {:?}", &self.ca[w_i.ck]);
            // See if we can skip this clause; if blocker already assigned, we don't care
            if self.value(w_i.blocker) == LBool::True {
                i += 1;
                j += 1;
                continue;
            }

            // For invariant, make sure false lit is second value in clause; we // want the first lit always to be unassigned (and the false lit is // assigned by definition from it being on the trail)
            let fl = !l;
            let ck = w_i.ck;
            let (first, c_sz) = {
                let c = &mut self.ca[ck];
                if c[0] == fl {
                    c[0] = c[1];
                    c[1] = fl;
                }
                i += 1;
                (c[0], c.size)
            };

            // If first watch is true, then this clause is sat already
            let w_first = Watcher::new(ck, first);
            if first != w_first.blocker && self.value(first) == LBool::True {
                watchers[j] = w_first;
                j += 1;
                continue;
            }

            // Find new watcher, since false has been assigned
            for k in 2..c_sz {
                // If not false, update that watcher's list too, and continue
                let c_k = self.ca[ck][k];
                if self.value(c_k) != LBool::False {
                    let c = &mut self.ca[ck];
                    c[1] = c[k];
                    c[k] = fl;
                    self.watches.get_watchers(!c[1]).push(w_first);
                    break 'next_watcher;
                }
            }

            watchers[j] = w_first;
            j += 1;
            // If we didn't find a new watcher, it's a unit clause: check if either
            // conflict, or a new implication
            if self.value(first) == LBool::False {
                conflict = Some(ck);
                // Copy remaining watches over, in case we've skipped some watches when we find new
                // watchers; this also has the effect of breaking out of the loop, since i == n_ws
                while i < n_ws {
                    watchers[j] = watchers[i];
                    i += 1;
                    j += 1;
                }
            } else {
                // If no conflict, we got a new implication, so add to trail and assign it
                self.add_to_trail(first, Some(ck));
            }
        }

        // Truncate watcher's list to remove any we've skipped to find new watchers
        watchers.truncate(j);
        self.watches.set_watchers(l, watchers);

        // Report if conflict occurred
        conflict
    }

    // Simplifies the clause database by removing satisfied clauses.
    // Returns whether the format is either unknown (true), or UNSAT (false).
    fn simplify(&mut self) -> bool {
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
        let mut next = V_UNDEF;
        // TODO: implement random decisions

        // Iterate until either heap empty, or non-assigned next var found
        while next == V_UNDEF || self.assigned[next as usize] != LBool::Undef {
            // if empty, mark next as undefined again and leave
            if self.act_heap.is_empty() {
                next = V_UNDEF;
                break;
            }
            (next, _) = self.act_heap.pop_with_key().unwrap();
        }
        // If undef, denote as such
        if next == V_UNDEF {
            None
        } else {
            // TODO: implement random polarity
            // Get polarity from previous decisions
            let polarity = self.polarity[next as usize];
            Some(Lit::new(next, polarity))
        }
    }

    // Analyze conflicts if one occurs. Returns the literals of the new clause, and the decision
    // level to which to backtrack.
    fn analyze_conflicts(&mut self, ck: ClauseKey) -> (Vec<Lit>, DecisionLevel) {
        // Record the current asserting literal.
        let mut a_lit = None;
        let mut vars_to_bump = Vec::with_capacity(16);
        self.learnt_lits.clear();
        self.seen_to_clear.clear();

        let mut learnt = {
            let c = &self.ca[ck];
            self.reason_lits.resize(c.lits.len(), Lit::default());
            self.reason_lits.copy_from_slice(&c.lits);
            c.learnt
        };
        // Loop until we've reverse-BFS'ed to the first UIP.
        let mut trail_ctr = 0;
        let mut rescale_clauses = false;
        loop {
            // If learnt, bump activity to prioritize in clause deletion
            if learnt {
                rescale_clauses = {
                    // Limit scope here to prevent borrow checker from angies
                    let c = &mut self.ca[ck];
                    c.bump_activity(self.cd_conf.inc_var, self.cd_conf.rescale_lim)
                };
            }

            // For every clause literal except the first, if not yet seen, bump its activity
            // and either increase counter if at/above current DL (since this is another node
            // we need to reverse-BFS), or add to learned literals.
            // TODO: why not the first, except when None?
            let s_idx = if a_lit.is_none() { 0 } else { 1 };
            for lit in self.reason_lits[s_idx..].into_iter() {
                let var = lit.var();
                // Skip base-DL lits
                let lvl = self.level(var);
                if lvl > 0 {
                    if !self.seen[lit.var_idx()] {
                        vars_to_bump.push(var);
                        self.seen[lit.var_idx()] = true;
                        if lvl >= self.decision_level {
                            trail_ctr += 1;
                        } else {
                            self.learnt_lits.push(*lit);
                        }
                    }
                }
            }

            // After processing each literal within the reason clause, find next clause to parse
            loop {
                let l = self.pop_trail();
                // If not seen yet, need to handle:
                if !self.seen[l.var_idx()] {
                    // Update asserting lit
                    a_lit = Some(l);
                    // Get new reason
                    let ck = self.reason_ref_mut(l.var()).ck.unwrap();
                    learnt = {
                        let c = &self.ca[ck];
                        self.reason_lits.resize(c.lits.len(), Lit::default());
                        self.reason_lits.copy_from_slice(&c.lits);
                        c.learnt
                    };
                    // TODO: do I need this?
                    self.seen[l.var_idx()] = false;
                    trail_ctr -= 1;
                    // Process a_lit
                    break;
                }
            }

            // Check if reached UIP
            if trail_ctr == 0 {
                break;
            }
        }

        if rescale_clauses {
            self.rescale_clause_activity();
        }
        for var in &vars_to_bump {
            self.bump_var_activity(*var);
        }

        // Set first learnt lit to negation of asserting lit (since we want to prevent this
        // implication in the future).)
        // TODO: why do we need the first to be asserting again? I think it's for watched lits and
        // CCM, but not entirely sure
        self.learnt_lits[0] = !a_lit.unwrap();

        // Mark seen vars to be cleared
        for l in &self.learnt_lits {
            self.seen_to_clear.push(l.var());
        }
        // TODO: implement ccm here later

        // Find backtrack level: if unit learnt lit, backtrack back to base DL to propagate
        let bt_lvl = if self.learnt_lits.len() == 1 {
            0
        } else {
            // Otherwise, find highest DL in learnt lits (that's not the asserting lit)
            let (max_i, max_lvl) = self.learnt_lits[1..]
                .into_iter()
                .enumerate()
                .max_by(|(_, l1), (_, l2)| self.level(l1.var()).cmp(&self.level(l2.var())))
                .map(|(i, l)| (i, self.level(l.var())))
                .unwrap();
            // Swap 2nd learned lit with the max level lit
            self.learnt_lits.swap(1, max_i);
            max_lvl
        };

        // Clear seen variables
        for v_to_clear in &self.seen_to_clear {
            self.seen[*v_to_clear as usize] = false;
        }
        self.seen_to_clear.clear();

        (self.learnt_lits.clone(), bt_lvl)
    }

    fn conflict_clause_minimization(&mut self, lits: &mut Vec<Lit>) {
        todo!()
    }

    // Backtrack to the desired decision level, cleaning up the trail as necessary.
    fn backtrack(&mut self, dl: DecisionLevel) {
        if self.decision_level <= dl {
            return;
        }

        // Get level delimiter trail index for this level
        let lvl_dlm_idx = self.trail.dl_delim_idx(dl);
        while self.trail_size() > lvl_dlm_idx {
            let lit = self.pop_trail();
            let var = lit.var_idx();
            // Unassign this variable
            self.assigned[var] = LBool::Undef;
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
        self.trail.dl_delim_idxs.truncate((dl + 1) as usize);
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
            }
            _ => {
                // Add to watchlist and slotmap
                let ck = self.create_clause(&lits, false);
                self.clauses.push(ck);
                self.attach_clause(ck);
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
        assert!(c_sz > 1);
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

    /// Auxiliary structure methods
    ///
    /// Add to the trail, updating the associated reason, dl, and assignment
    fn add_to_trail(&mut self, lit: Lit, ck: Option<ClauseKey>) {
        assert!(self.value(lit) == LBool::Undef);
        // If +Lit (i.e. sign == false), assign true; otherwise, assign false
        self.assigned[lit.var_idx()] = LBool::from(!lit.sign() as u8);
        // Add reason to antecedent graph, and push to trail
        self.reasons[lit.idx()] = Reason {
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

    /// Literal/Variable accesses
    ///
    /// Calculate value
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
    fn reason_ref_mut(&mut self, v: Var) -> &mut Reason {
        &mut self.reasons[v as usize]
    }

    fn n_vars(&self) -> usize {
        self.var_mapping.len()
    }

    /// Statistics computations for activity, LBD, etc.
    ///
    /// Bumps the activity of the clause. Returns if clause scaling is needed.
    fn bump_clause_activity(&self, c: &mut Clause) -> bool {
        c.act += self.cd_conf.inc_var;

        // If exceeds limit, let caller know
        c.act >= self.cd_conf.rescale_lim
    }

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
        let lim = F64::new(self.dh_conf.rescale_lim).unwrap();
        if self.acts[var] >= lim {
            for a in &mut self.acts {
                *a *= self.dh_conf.rescale_f;
            }
            self.dh_conf.inc_var *= self.dh_conf.rescale_f;
            self.rebuild_heap();
        }

        // Update heap
        *self.act_heap.get_mut(&v).unwrap() = self.acts[var];
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

    fn clause_iter(&mut self) -> Iter<'_, ClauseKey, Clause> {
        self.ca.iter()
    }

    fn clause_iter_mut(&mut self) -> IterMut<'_, ClauseKey, Clause> {
        self.ca.iter_mut()
    }

    /// All propagated iff bcp_idx >= trail.size()
    fn all_propagated(&self) -> bool {
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
        self.trail.trail[idx]
    }
}
