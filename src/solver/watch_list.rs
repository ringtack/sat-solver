use std::mem;

use super::{
    clause::ClauseKey,
    types::Lit,
    util::{remove, vec_with_size},
};

pub struct WatchList {
    // Literal -> List of Watchers (i.e. clauses in which this Lit is watched)
    occs: Vec<Vec<Watcher>>,
}

impl WatchList {
    // Creates a watch list for n variables.
    pub fn new(n_lits: usize) -> Self {
        Self {
            occs: vec_with_size(n_lits, vec![]),
        }
    }

    // Adds a watcher to the literal's watched clauses list.
    pub fn add_watcher(&mut self, l: Lit, w: Watcher) {
        self.occs[l.v as usize].push(w);
    }

    // Removes a watcher to the literal's watched clauses list, if it exists.
    pub fn remove_watcher(&mut self, l: Lit, w: Watcher) {
        remove(&mut self.occs[l.v as usize], w);
    }

    // Get a mutable reference to the literal's watch list.
    pub fn get_watchers(&mut self, l: Lit) -> &mut Vec<Watcher> {
        &mut self.occs[l.v as usize]
    }

    /// Hands ownership of this specific watchers to the caller. Make sure to put it back with
    /// set_watchers.
    pub fn take_watchers(&mut self, l: Lit) -> Vec<Watcher> {
        mem::take(&mut self.occs[l.v as usize])
    }

    pub fn set_watchers(&mut self, l: Lit, ws: Vec<Watcher>) {
        self.occs[l.v as usize] = ws;
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub struct Watcher {
    pub ck: ClauseKey,
    pub blocker: Lit,
}

impl Watcher {
    pub fn new(ck: ClauseKey, blocker: Lit) -> Self {
        Self { ck, blocker }
    }
}

/*
impl WatchList {
    // Creates a watch list for n variables.
    pub fn new(n_lits: usize) -> Self {
        Self {
            occs: vec_with_size(n_lits, RefCell::new(vec![])),
        }
    }

    // Adds a watcher to the literal's watched clauses list.
    pub fn add_watcher(&mut self, l: Lit, w: Watcher) {
        self.occs[l.v as usize].borrow_mut().push(w);
    }

    // Removes a watcher to the literal's watched clauses list, if it exists.
    pub fn remove_watcher(&mut self, l: Lit, w: Watcher) {
        let mut x = self.occs[l.v as usize].borrow_mut();
        remove(&mut *x, w);
    }

    // Get a mutable reference to the literal's watch list.
    pub fn get_watchers(&mut self, l: Lit) -> RefMut<'_, Vec<Watcher>> {
        self.occs[l.v as usize].borrow_mut()
    }
}
 */
