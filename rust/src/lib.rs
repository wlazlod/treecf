//! treecf Rust core (dev-only until the benchmark gate).

pub mod constraints;
pub mod ir;

#[cfg(feature = "python")]
mod py;
