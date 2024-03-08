use std::fmt::Debug;

use fxhash::FxHashSet;

#[derive(Clone)]
pub struct SATInstance {
    pub n_vars: usize,
    pub n_clauses: usize,
    pub clauses: Vec<Clause>,
    // (Positive) list of all variables in the instance
    pub vars: FxHashSet<Variable>,
}

impl Debug for SATInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "n_vars: {}\tn_clauses: {}", self.n_vars, self.n_clauses).unwrap();
        for c in &self.clauses {
            write!(f, "Clause:").unwrap();
            for l in &c.lits {
                write!(f, " {l}").unwrap();
            }
            writeln!(f).unwrap();
        }
        writeln!(f)
    }
}

#[derive(Clone)]
pub struct Clause {
    pub lits: Vec<Literal>,
}

pub type Literal = i64;
pub type Variable = i64;
