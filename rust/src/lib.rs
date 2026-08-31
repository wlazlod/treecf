//! treecf Rust core: the search engines behind the Python API.

#![forbid(unsafe_code)]

pub mod cells;
pub mod constraints;
pub mod exact;
pub mod ga;
pub mod interrupt;
pub mod ir;
pub mod regions;

#[cfg(feature = "python")]
mod py;
