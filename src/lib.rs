#![warn(
    clippy::pedantic,
    clippy::suspicious,
    clippy::perf,
    clippy::complexity,
    clippy::style,
    clippy::cargo,
    clippy::imprecise_flops
)]

//! Optimisers for use with the candle framework for lightweight machine learning.
//! These currently all implement the [`candle_nn::optim::Optimizer`] trait from candle-nn
pub mod adadelta;
pub mod adagrad;
pub mod adam;
pub mod adamax;
pub mod esgd;
pub mod lbfgs;
pub mod nadam;
pub mod radam;
pub mod rmsprop;
