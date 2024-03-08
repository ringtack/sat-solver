#[derive(Clone, Debug, Default)]
pub struct RuntimeStats {
    /// Record total (i.e. monotonically increasing) number of:
    /// - solves: number of solve attempts.
    /// - starts: number of (re)starts.
    /// - decisions: number of decisions made.
    /// - rand_decisions: number of rand_decisions made.
    /// - propagations: number of propagations made.
    /// - conflicts: number of conflicts that occur
    /// - deletions: number of clause deletions performed
    pub solves: u64,
    pub starts: u64,
    pub decisions: u64,
    pub rand_decisions: u64,
    pub propagations: u64,
    pub conflicts: u64,
    pub deletions: u64,

    /// Record current values of:
    /// - n_clauses: num og constraint clauses
    /// - n_clause_lits: num lits in constraint clauses
    /// - n_learnts: num learnt clauses
    /// - n_learnt_lits: num lits in learnt clauses
    /// - tot_lits: total literals learnt
    /// - max_lits: max possible literals learnt (before CCM)
    /// TODO: figure out how removing / adding constraint clauses works
    pub n_clauses: u64,
    pub n_clause_lits: u64,
    pub n_learnts: u64,
    pub n_learnt_lits: u64,
    pub tot_lits: u64,
    pub max_lits: u64,
}
