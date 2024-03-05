use std::collections::HashSet;

pub struct SATInstance {
    pub num_vars: usize,
    pub num_clauses: usize,
    pub clauses: Vec<Clause>,
}

pub struct Clause {
    pub literals: HashSet<Literal>,
}

pub struct Literal {
    pub var: usize,
    pub val: bool,
}
