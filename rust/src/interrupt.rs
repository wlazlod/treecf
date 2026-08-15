//! The stop question a long-running search polls, and the answer shape it
//! reports back with.
//!
//! A probe is a caller-supplied closure that takes no arguments, reads no
//! search state and returns no value beyond "stop now?". A search polls it on
//! a schedule fixed by its own work counters — never on anything derived from
//! the data it is searching — and the only thing a `true` answer can do is
//! abandon the search. So a probe that always answers `false` leaves every
//! number a search computes exactly as it was before the probe existed.

/// The stop question: `true` means abandon the search.
///
/// It is a `FnMut` because the real one carries state of its own (the pending
/// error a caller wants to raise afterwards); nothing here depends on that.
pub type InterruptProbe<'a> = &'a mut dyn FnMut() -> bool;

/// What a probed search answers with.
///
/// `Interrupted` deliberately carries nothing. Whatever the search had found
/// when the probe fired — an incumbent, a half-grown box, the results of the
/// chunks already done — is dropped here, so no caller can mistake an
/// abandoned search for a finished one.
#[derive(Debug)]
pub enum SearchOutcome<T> {
    Done(T),
    Interrupted,
}
