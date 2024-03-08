use super::types::{DecisionLevel, Lit};

// Assignment trail during search and inference.
pub struct AssignmentStack {
    // Stack of Lits
    // Assignment trail (either from decision, or BCP)
    pub trail: Vec<Lit>,
    // lvl -> index into trail
    // Indices for decision level delimiters.
    // (i.e. after deciding lvl, dl_delim_idxs[lvl] == new decided var location on new level)
    pub dl_delim_idxs: Vec<usize>,
    // Index from which to start BCP
    // - Updated in decide, and on clause add if unit clause detected
    pub bcp_idx: usize,
}

impl AssignmentStack {
    pub fn new(n_vars: usize) -> Self {
        Self {
            trail: Vec::with_capacity(n_vars),
            dl_delim_idxs: Vec::new(),
            bcp_idx: 0,
        }
    }

    // Pushes a lit onto the trail.
    pub fn push(&mut self, l: Lit) {
        self.trail.push(l);
    }

    // Gets the Lit in the trail at the index.
    pub fn get(&self, i: usize) -> Lit {
        self.trail[i]
    }

    // Gets the Lit at the current BCP index, then increments it.
    pub fn get_next_bcp_lit(&mut self) -> Option<Lit> {
        if self.bcp_idx >= self.trail.len() {
            None
        } else {
            let lit = self.trail[self.bcp_idx];
            self.bcp_idx += 1;
            Some(lit)
        }
    }

    // Sets BCP index to new value.
    pub fn set_bcp_idx(&mut self, v: usize) {
        self.bcp_idx = v;
    }

    // Sets the BCP index up to the trail head.
    pub fn set_bcp_idx_to_trail_head(&mut self) {
        self.bcp_idx = self.trail.len();
    }

    /// Checks if BCP index at end (i.e. all propagated).
    pub fn bcp_idx_at_end(&self) -> bool {
        self.bcp_idx >= self.trail.len()
    }

    /// Gets the delim index within the trail for the specified level.
    pub fn dl_delim_idx(&self, dl: DecisionLevel) -> usize {
        self.dl_delim_idxs[dl as usize]
    }
}
