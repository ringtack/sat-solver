use log;
use ringbuf::HeapRb;

use super::types::LBD;

// Restart policy configs.
pub const LUBY_DEFAULT: RestartPolicy = RestartPolicy::Luby(256);
pub const GLUCOSE_DEFAULT: RestartPolicy = RestartPolicy::Glucose(50, 0.8, 5000, 1.4);

// Clause deletion configs.
pub const KEEP_F_DEFAULT: f64 = 0.75; // from satch
pub const DELETION_SORT_ORDER_DEFAULT: [DeletionSortOption; 3] = [
    LBD_DELETION_DEFAULT,
    ACTIVITY_DELETION_DEFAULT,
    DeletionSortOption::ClauseSize,
];
pub const LBD_DELETION_DEFAULT: DeletionSortOption = DeletionSortOption::LBD(2000, 300);
pub const ACTIVITY_DELETION_DEFAULT: DeletionSortOption =
    DeletionSortOption::Activity(1.0, 1.0 / 0.999, 20);

// Decision policy configs.
pub const VSIDS_DEFAULT: HeuristicOption = HeuristicOption::VSIDS(256, 2);
pub const EVSIDS_DEFAULT: HeuristicOption = HeuristicOption::EVSIDS(1.0, 1. / 0.95, 100);

pub struct SolverConfig {
    /// Log filter. Set using `log::set_max_level().`
    pub verbosity: log::LevelFilter,
    /// Initial scaling factor for max learnt clauses relative to # clauses (default 1/3)
    pub max_learnt_f: f64,

    /// Whether to phase save on backtrack.
    pub save_phases: bool,
    /// Whether to remove satisfied constraint clauses
    pub remove_satisfied: bool,

    // Restart, clause deletion, decision, etc. policies
    pub restart_policy: RestartPolicy,
    pub deletion_policy: ClauseDeletionPolicy,
    pub decision_policy: DecisionPolicy,
    // TODO: conflict clause minimization (subsumption), phase saving
}

impl SolverConfig {
    pub fn opt_config(&self) -> OptConfig {
        OptConfig {
            save_phases: self.save_phases,
            remove_satisfied: self.remove_satisfied,
        }
    }

    pub fn clause_deletion_config(&self) -> ClauseDeletionConfig {
        let mut cdc = ClauseDeletionConfig::default();
        // Search self for deletion policies
        for opt in &self.deletion_policy.sort_order {
            match *opt {
                DeletionSortOption::LBD(u, scale) => {
                    cdc.u = u;
                    cdc.k = scale;
                }
                DeletionSortOption::Activity(inc_var, f, exp) => {
                    cdc.inc_var = inc_var;
                    cdc.f = f;
                    cdc.rescale_lim = 1. * 10_f64.powi(exp);
                    cdc.rescale_f = 1. * 10_f64.powi(-exp);
                }
                DeletionSortOption::ClauseSize => (),
            }
        }
        cdc
    }

    pub fn restart_config(&self) -> RestartConfig {
        let mut rc = RestartConfig::default();
        match self.restart_policy {
            RestartPolicy::Luby(u) => {
                rc.u = u as usize;
            }
            RestartPolicy::Glucose(x_lbd_win, k, x_ass_win, r) => {
                rc.lbd_win = HeapRb::new(x_lbd_win as usize);
                rc.k = k;
                rc.ass_win = HeapRb::new(x_ass_win as usize);
                rc.r = r;
            }
            RestartPolicy::Rapid => unimplemented!("TODO: implement time permitting"),
        }
        rc
    }

    pub fn decision_config(&self) -> DecisionConfig {
        let mut hc = DecisionConfig::default();
        match self.decision_policy.heuristic {
            HeuristicOption::VSIDS(_, _) => unimplemented!(),
            HeuristicOption::EVSIDS(inc_var, f, exp) => {
                hc.inc_var = inc_var;
                hc.f = f;
                hc.rescale_lim = 1. * 10_f64.powi(exp);
                hc.rescale_f = 1. * 10_f64.powi(-exp);
            }
            HeuristicOption::CHB => unimplemented!(),
        };
        hc
    }
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            // Tuning params
            verbosity: log::max_level(),
            max_learnt_f: 1. / 3.,

            save_phases: true,
            remove_satisfied: true,

            // Policies
            restart_policy: GLUCOSE_DEFAULT,
            deletion_policy: ClauseDeletionPolicy {
                keep_bin_clauses: false,
                keep_rec_clauses: false,
                keep_f: KEEP_F_DEFAULT,
                sort_order: DELETION_SORT_ORDER_DEFAULT.to_vec(),
            },
            decision_policy: DecisionPolicy {
                heuristic: EVSIDS_DEFAULT,
                random_var: None,
                random_pol: false,
            },
        }
    }
}

// Config options for the restart policy.
#[derive(Clone, Debug)]
pub enum RestartPolicy {
    /// u: scaling factor
    Luby(u64),
    /// X_rec_lbd: window size of recent learnt LBDs
    /// K: scaling factor to check if X_rec_lbd too large
    /// X_rec_ass: window size of recent values #(assigned literals during conflicts)
    /// R: scaling factor to check if X_rec_ass too large
    Glucose(u64, f64, u64, f64),
    // TODO: implement if possible
    Rapid,
}

// Config options for clause deletion.
#[derive(Clone, Debug)]
pub struct ClauseDeletionPolicy {
    // Whether to always keep binary clauses
    pub keep_bin_clauses: bool,
    // Whether to always keep recent clauses
    pub keep_rec_clauses: bool,
    // How many clauses to keep [0, 1].
    pub keep_f: f64,
    // Sort order; must be non-negative.
    pub sort_order: Vec<DeletionSortOption>,
}

// Deletion configs
#[derive(Clone, Debug)]
pub enum DeletionSortOption {
    /// u: baseline number of conflicts required.
    /// scale: how much to increase baseline for each restart.
    LBD(u64, u64),
    /// inc_var: starting increase value.
    /// f: scaling factor for inc_var (i.e. inc_var * f every increment). MiniSat scales by 1/f,
    ///    but we compute before.
    /// rescale_lim_exp: limit before we re-scale every activity value (after which we multiply by
    ///    -rescale_lim). Note that this is in exponent form, i.e. rescale_lim_exp=10 -> 1e10.
    /// NOTE: this is for learnt clause deletion, not decision heuristics.
    Activity(f64, f64, i32),
    /// Sort by clause size
    ClauseSize,
}

// Config options for decision heuristics.
#[derive(Clone, Debug)]
pub struct DecisionPolicy {
    // Branching heuristic to use
    heuristic: HeuristicOption,
    // Whether to randomly select branching literal (if so, the frequency), or its polarity.
    // default false for both
    random_var: Option<f64>,
    random_pol: bool,
}

// Heuristic configs
#[derive(Clone, Debug)]
pub enum HeuristicOption {
    /// rescore_freq: frequency of rescoring activities.
    /// decay: decay factor.
    VSIDS(u64, u64),
    /// inc_var: starting increase value.
    /// f: scaling factor for inc_var (i.e. inc_var * f). See DeletionSortOption::Activity.
    /// rescale_lim_exp: limit before we re-scale every activity value (after which we multiply by
    ///    -rescale_lim). Note that this is in exponent form, i.e. rescale_lim_exp=10 -> 1e10.
    /// NOTE: this is for per-variable decision heuristics.
    EVSIDS(f64, f64, i32),
    // TODO: implement if time permitting
    CHB,
}

// Options (i.e. flags, params, idk man) config
#[derive(Default, Clone, Copy, Debug)]
pub struct OptConfig {
    /// Whether to phase save on backtrack.
    pub save_phases: bool,
    /// Whether to remove satisfied constraint clauses
    pub remove_satisfied: bool,
}

// Decision heuristics config.
#[derive(Default, Clone, Copy, Debug)]
pub struct DecisionConfig {
    // TODO: do VSIDS stuff time permitting
    /// EVSIDS policy values.
    ///
    /// Increase value on increment.
    pub inc_var: f64,
    /// Scaling factor for inc_var.
    pub f: f64,
    /// Limit before we re-scale every clause's activity value by f
    pub rescale_lim: f64,
    pub rescale_f: f64,
}

// Clause deletion config.
#[derive(Default, Clone, Copy, Debug)]
pub struct ClauseDeletionConfig {
    /// Glucose (LBD) based clause deletion policy values
    ///
    /// Baseline number of conflicts required before clause deletion.
    pub u: u64,
    /// How much to increase baseline for each restart.
    pub k: u64,

    /// Activity based clause deletion policy values.
    ///
    /// Increase value on increment.
    pub inc_var: f64,
    /// Scaling factor for inc_var.
    pub f: f64,
    /// Limit before we re-scale every clause's activity value by f
    pub rescale_lim: f64,
    pub rescale_f: f64,
}

// Restart config.
pub struct RestartConfig {
    /// Luby related restart values.
    ///
    /// Scaling factor for each Luby restart value.
    pub u: usize,

    /// Glucose related restart values.
    ///
    /// Sliding window of last learnt LBDs, and its corresponding avg.
    pub lbd_win: HeapRb<LBD>,
    pub avg_lbd_win: f64,
    /// Global average of learnt LBDs.
    pub avg_lbd: f64,
    /// Scale factor to check if recent average too large (i.e. avg_lbd_win * k > avg_lbd)
    pub k: f64,
    /// Sliding window of number of assigned literals during conflicts, and its corresponding avg.
    pub ass_win: HeapRb<usize>,
    pub avg_ass_win: f64,
    /// Global average of number of assigned literals during conflicts.
    pub avg_ass: f64,
    /// Scale factor to check if recent average too large (i.e. avg_ass_win > R * avg_ass)
    pub r: f64,
}

impl Default for RestartConfig {
    fn default() -> Self {
        Self {
            u: 0,
            lbd_win: HeapRb::new(1),
            avg_lbd_win: 0.,
            avg_lbd: 0.,
            k: 0.,
            ass_win: HeapRb::new(1),
            avg_ass_win: 0.,
            avg_ass: 0.,
            r: 0.,
        }
    }
}
