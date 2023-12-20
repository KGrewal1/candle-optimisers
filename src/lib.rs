/*!
Optimisers for use with the candle framework for lightweight machine learning.
Apart from LBFGS, these all implement the [`candle_nn::optim::Optimizer`] trait from candle-nn
*/

use std::fmt::Debug;

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

/// Trait for optimisers to expose their parameters
pub trait OptimParams: candle_nn::optim::Optimizer {
    /// get the current parameters of the Optimiser
    fn params(&self) -> &Self::Config;
    /// set the current parameters of the Optimiser
    fn set_params(&mut self, config: Self::Config);
}

/// Trait for Models: this is needed for optimisers that require the ability to calculate the loss
/// such as LBFGS
pub trait Model: Sized {
    /// get the loss of the model
    fn loss(&self) -> CResult<Tensor>; //, xs: &Tensor, ys: &Tensor
}

/// trait for optimisers like LBFGS that need the ability to calculate the loss
/// and its gradient
pub trait LossOptimizer<M: Model>: Sized {
    /// type of the optimiser configuration
    type Config: Sized;
    /// create a new optimiser from a Vec of variables, setup parameters and a model
    fn new(vs: Vec<Var>, params: Self::Config, model: M) -> CResult<Self>;
    /// take a step of the optimiser
    fn backward_step(&mut self, loss: &Tensor) -> CResult<ModelOutcome>; //, xs: &Tensor, ys: &Tensor
    /// get the current learning rate
    fn learning_rate(&self) -> f64;
    /// set the learning rate
    fn set_learning_rate(&mut self, lr: f64);
    /// get the a vec of the variables being optimised
    fn into_inner(self) -> Vec<Var>;
    /// create a new optimiser from a slice of variables, setup parameters and a model
    fn from_slice(vars: &[&Var], config: Self::Config, model: M) -> CResult<Self> {
        let vars: Vec<_> = vars.iter().map(|&v| v.clone()).collect();
        Self::new(vars, config, model)
    }
}

/// Outcomes of an optimiser step for methods such as LBFGS
#[derive(Debug)]
pub enum ModelOutcome {
    /// The model took a step and the loss decreased
    /// contains next loss and the number of func evals
    Stepped(Tensor, usize),
    /// The model has converged and the loss has not changed
    /// contains loss and the number of func evals
    Converged(Tensor, usize),
}

/// Method of weight decay to use
#[derive(Clone, Copy, Debug)]
pub enum Decay {
    /// Weight decay regularisation to penalise large weights
    ///
    /// The gradient is transformed as
    /// $$ g_{t} \\gets g_{t} + \\lambda  \\theta_{t-1}$$
    ///
    /// This is equivalent to an L2 regularisation term in the loss adding $\\frac{\\lambda}{2}||\theta||_{2}^{2}$ but avoids autodifferentiation
    /// of the L2 term
    WeightDecay(f64),
    /// Decoupled weight decay as described in [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
    ///
    /// This directly decays the weights as
    ///
    /// $$ \\theta_{t} \\gets (1 - \\eta \\lambda) \\theta_{t-1}$$
    ///
    /// This is equivalent to regularisation, only for SGD without momentum, but is different for adaptive gradient methods
    DecoupledWeightDecay(f64),
}

/// Type of momentum to use
#[derive(Copy, Clone, Debug)]
pub enum Momentum {
    /// classical momentum
    Classical(f64),
    /// nesterov momentum
    Nesterov(f64),
}
