//! treecf Rust core (dev-only until the benchmark gate).

pub mod cells;
pub mod constraints;
pub mod exact;
pub mod ga;
pub mod ir;
pub mod regions;

#[cfg(feature = "python")]
mod py;
