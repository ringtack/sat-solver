use std::fmt::Display;
use std::ops::{BitAnd, BitXor, Not, Shr};

use ordered_float::NotNan;

/// Representations for LBD, DL, etc. so I'm consistent
pub type LBD = u16;
pub type DecisionLevel = u16;

/// Let us use f64s as Ord
pub type F64 = NotNan<f64>;

/// Representation of a variable, so I don't forget
pub type Var = i64;
// Since we won't have any negative variables
pub const V_UNDEF: Var = -1;

/// How to compute n lits from v vars? Given v vars, n = v * 2. This works for indexing, since
/// our first variable starts at 0.
pub fn lits_from_vars(n_vars: usize) -> usize {
    n_vars * 2
}

/// Representation of a Literal, using the MiniSat convention: lit.v = 2 * var + sign
#[derive(Hash, Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Lit {
    pub v: i64,
}

// Custom markers for Lits; since we scale the literals up, and thus use no negatives, there
// should be no conflicts here.
pub const L_UNDEF: Lit = Lit { v: -2 };
pub const _L_ERROR: Lit = Lit { v: -1 };

impl Lit {
    // Here, a TRUE sign == NEGATIVE
    pub fn new(v: Var, sign: bool) -> Lit {
        Lit {
            v: v + v + (sign as i64),
        }
    }

    // Returns true if sign is negative.
    pub fn sign(&self) -> bool {
        self.v.bitand(1) != 0
    }

    pub fn var(&self) -> Var {
        self.v.shr(1)
    }

    // Variable, but cast as usize to index
    pub fn var_idx(&self) -> usize {
        self.v.shr(1) as usize
    }

    // Get v as an index
    #[inline(always)]
    pub fn idx(&self) -> usize {
        self.v as usize
    }
}

impl Not for Lit {
    type Output = Self;
    fn not(self) -> Lit {
        Self {
            v: self.v.bitxor(1),
        }
    }
}

impl Display for Lit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", if self.sign() { "-" } else { "" }, self.var())
    }
}

// Represent false, true, or UNDEF (i.e. not yet assigned). We prefer this over an Option<Bool>,
// since each option requires a pointer, whereas we only really have 3 values (i.e. u8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum LBool {
    True = 0,  // 0
    False = 1, // 1
    Undef = 2, // 2
}

impl LBool {
    pub fn from_sign(s: bool) -> LBool {
        // TODO: verify !s, not s
        LBool::from(!s as u8)
    }
}

impl From<LBool> for bool {
    #[inline(always)]
    fn from(value: LBool) -> Self {
        match value {
            LBool::True => true,
            _ => false,
        }
    }
}

impl From<u8> for LBool {
    #[inline(always)]
    fn from(value: u8) -> Self {
        match value {
            0 => Self::True,
            1 => Self::False,
            _ => Self::Undef,
        }
    }
}

impl Default for LBool {
    fn default() -> Self {
        Self::Undef
    }
}

impl BitXor for LBool {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> LBool {
        LBool::from((self as u8).bitxor(rhs as u8))
    }
}

// TODO: pretty sure default impl is good enough?
// impl PartialEq for LBool {
//   fn eq(&self, other: &Self) -> bool {
//     (self as u8 == 2 && other as u8 == 2) ||
//       (other as u8!= 2 && (self as u8 == other as u8))
//   }

//   fn ne(&self, other: &Self) -> bool {
//     self as u8 != other as u8
//   }
// }

// Status markers
#[derive(Debug, Clone, Copy)]
pub enum SolveStatus {
    Unknown,
    SAT,
    UNSAT,
}

// #[derive(Clone, Debug, Copy)]
// pub enum BCPStatus {
//     CONFLICT,
//     SAT,
//     UNKNOWN,
// }

// pub struct SolveResult;
