//! treecf Rust core (dev-only until the benchmark gate).

pub mod cells;
pub mod constraints;
pub mod ga;
pub mod ir;

#[cfg(feature = "python")]
mod py;
