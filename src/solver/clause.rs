use std::fmt::Debug;
use std::ops::{Index, IndexMut};

use super::types::{DecisionLevel, Lit, LBD};
use slotmap::{
    self,
    basic::{Iter, IterMut},
    new_key_type, Key, SlotMap,
};

// Note that default is ClauseKey::null()
new_key_type! {
  pub struct ClauseKey;
}

#[derive(Default)]
pub struct ClauseAllocator {
    sm: SlotMap<ClauseKey, Clause>,
}

impl ClauseAllocator {
    pub fn new(n_clauses: usize) -> Self {
        Self {
            sm: SlotMap::with_capacity_and_key(n_clauses),
        }
    }

    // Create a new clause from the provided literals.
    pub fn create_clause(&mut self, lits: &[Lit], learnt: bool) -> ClauseKey {
        let ck = self
            .sm
            // Insert clause with the generated clause key
            .insert_with_key(|ck| Clause::with_key(lits, learnt, ck));
        // Return clause key
        ck
    }

    pub fn iter(&self) -> Iter<'_, ClauseKey, Clause> {
        self.sm.iter()
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, ClauseKey, Clause> {
        self.sm.iter_mut()
    }

    // pub fn get(&self, i: ClauseKey) -> &Clause {
    //     self.sm.get(i).unwrap()
    // }
}

impl Index<ClauseKey> for ClauseAllocator {
    type Output = Clause;
    fn index(&self, index: ClauseKey) -> &Self::Output {
        // TODO: I probably need to check this...
        self.sm.get(index).unwrap()
    }
}

impl IndexMut<ClauseKey> for ClauseAllocator {
    fn index_mut(&mut self, index: ClauseKey) -> &mut Self::Output {
        // TODO: I probably need to check this...
        self.sm.get_mut(index).unwrap()
    }
}

#[derive(Default, Clone)]
pub struct Clause {
    /// Size (# literals)
    pub size: usize,
    pub lits: Vec<Lit>,

    /// Header metadata
    /// TODO: see if I put this at the start or end of struct?
    ///
    /// Reference to this clause in the slotmap (i.e. clause key)
    pub ck: ClauseKey,
    /// TODO: maybe convert into bits later
    /// LBD (Glucose level)
    pub lbd: LBD,
    /// Activity
    pub act: f64,
    /// Whether clause was learnt
    pub learnt: bool,
    /// Whether clause is protected (i.e. just added this deletion cycle)
    pub protected: bool,
    /// TODO: (probably won't need) collect at next GC
    pub garbage: bool,
}

impl Clause {
    /// Create a new clause. Probably don't need to use this yourself, as ClauseAllocator should
    /// handle it for you.
    pub fn new(lits: &[Lit], learnt: bool) -> Self {
        Self {
            size: lits.len(),
            lits: lits.to_vec(),
            ck: ClauseKey::null(),
            lbd: 0,
            act: 0.,
            learnt,
            protected: false,
            garbage: false,
        }
    }

    /// Increases the clause's activity. Returns if the new activity exceeds the limit.
    pub fn bump_activity(&mut self, var_inc: f64, lim: f64) -> bool {
        self.act *= var_inc;
        self.act >= lim
    }

    fn with_key(lits: &[Lit], learnt: bool, ck: ClauseKey) -> Self {
        Self {
            size: lits.len(),
            lits: lits.to_vec(),
            ck,
            lbd: 0,
            act: 0.,
            learnt,
            protected: false,
            garbage: false,
        }
    }
}

impl Debug for Clause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lit_str = self
            .lits
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join(",");
        write!(
            f,
            "Clause {{ size: {}, learnt: {}, lits: {} }}",
            self.size, self.learnt, lit_str
        )
    }
}

impl Index<usize> for Clause {
    type Output = Lit;
    fn index(&self, i: usize) -> &Lit {
        &self.lits[i]
    }
}
impl IndexMut<usize> for Clause {
    fn index_mut(&mut self, i: usize) -> &mut Lit {
        &mut self.lits[i]
    }
}

// Record the reason and decision level for an implication (i.e. BCP result), if exists.
#[derive(Clone, Copy, Debug, Default)]
pub struct Reason {
    // Slot key for clause (if exists; e.g. for decisions, ck == None)
    pub ck: Option<ClauseKey>,
    // Decision level
    pub dl: DecisionLevel,
}

/*
impl ClauseAllocator {
    pub fn new(n_clauses: usize) -> Self {
        Self {
            sm: SlotMap::with_capacity_and_key(n_clauses),
        }
    }

    // Create a new clause from the provided literals.
    pub fn create_clause(&mut self, lits: &[Lit], learnt: bool) -> ClauseKey {
        let ck = self
            .sm
            // Insert clause with the generated clause key
            .insert_with_key(|ck| RefCell::new(Clause::with_key(lits, learnt, ck)));
        // Return clause key
        ck
    }

    pub fn iter(&self) -> Iter<'_, ClauseKey, RefCell<Clause>> {
        self.sm.iter()
    }

    pub fn get(&self, i: ClauseKey) -> &RefCell<Clause> {
        self.sm.get(i).unwrap()
    }
}
 */
