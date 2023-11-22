#![warn(
    clippy::pedantic,
    clippy::suspicious,
    clippy::perf,
    clippy::complexity,
    clippy::style,
    clippy::cargo,
    clippy::imprecise_flops
)]
#![allow(
    clippy::missing_errors_doc,
    clippy::too_many_lines,
    clippy::uninlined_format_args
)]

//! Optimisers for use with the candle framework for lightweight machine learning.
//! These currently all implement the [`candle_nn::optim::Optimizer`] trait from candle-nn

use candle_core::Result as CResult;
use candle_core::Tensor;
use candle_core::Var;
pub mod adadelta;
pub mod adagrad;
pub mod adam;
pub mod adamax;
pub mod esgd;
pub mod lbfgs;
pub mod nadam;
pub mod radam;
pub mod rmsprop;

/// trait for Models: this will be needed for optimisers that require the ability to calculate the loss
/// such as LBFGS
///
/// This is largely the same as the trait defined in the MNIST example in the main candle repo
pub trait Model: Sized {
    // initialise
    // fn new(vs: VarBuilder) -> CResult<Self>;
    // // forward pass through network
    // fn forward(&self, xs: &Tensor) -> CResult<Tensor>;
    // pass comparing to actual to get loss
    fn loss(&self) -> CResult<Tensor>; //, xs: &Tensor, ys: &Tensor
}

/// trait for optimisers like LBFGS that need the ability to calculate the loss
/// and its gradient
pub trait LossOptimizer<M: Model>: Sized {
    type Config: Sized;
    fn new(vs: Vec<Var>, params: Self::Config, model: M) -> CResult<Self>;
    fn backward_step(&mut self) -> CResult<()>; //, xs: &Tensor, ys: &Tensor
    fn learning_rate(&self) -> f64;
    fn set_learning_rate(&mut self, lr: f64);
    fn into_inner(self) -> Vec<Var>;
    fn empty(config: Self::Config, model: M) -> CResult<Self> {
        Self::new(vec![], config, model)
    }
    fn from_slice(vars: &[&Var], config: Self::Config, model: M) -> CResult<Self> {
        let vars: Vec<_> = vars.iter().map(|&v| v.clone()).collect();
        Self::new(vars, config, model)
    }
}
