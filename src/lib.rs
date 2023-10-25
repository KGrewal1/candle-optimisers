#![warn(
    clippy::pedantic,
    clippy::suspicious,
    clippy::perf,
    clippy::complexity,
    clippy::style,
    clippy::cargo
)]
pub mod adadelta;
pub mod adagrad;
pub mod adamax;
pub mod esgd;
pub mod nadam;
pub mod radam;
pub mod rmsprop;
