/*!
Optimisers for use with the [candle](https://github.com/huggingface/candle) framework for lightweight machine learning.
Apart from LBFGS, these all implement the [`candle_nn::optim::Optimizer`] trait from candle-nn

# Example

Training an MNIST model using the Adam optimiser

```
# use candle_core::{Result, Tensor};
# use candle_core::{DType, D};
# use candle_nn::{loss, ops, VarBuilder, VarMap, optim::Optimizer};
# use candle_optimisers::{
#     adam::{Adam, ParamsAdam}
# };
#
# pub trait Model: Sized {
#     fn new(vs: VarBuilder) -> Result<Self>;
#     fn forward(&self, xs: &Tensor) -> Result<Tensor>;
# }
#
# pub fn training_loop<M: Model>(
#     m: candle_datasets::vision::Dataset,
#     varmap: &VarMap,
#     model: M,
# ) -> anyhow::Result<()> {
#     // check to see if cuda device availabke
#     let dev = candle_core::Device::cuda_if_available(0)?;
#     // get the input from the dataset and put on device
#     let train_images = m.train_images.to_device(&dev)?;
#     // get the training labels on the device
#     let train_labels = m.train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
#
#
#     // load the test images
#     let test_images = m.test_images.to_device(&dev)?;
#     // load the test labels
#     let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
#
    // create the Adam optimiser

    // set the learning rate to 0.004 and use the default parameters for everything else
    let params = ParamsAdam {
            lr: 0.004,
            ..Default::default()
        };
    // create the optimiser by passing in the variable to be optimised and the parameters
    let mut optimiser = Adam::new(varmap.all_vars(),  params)?;

    // loop for model optimisation
    for epoch in 0..100 {
        // run the model forwards
        // get log probabilities of results
        let logits = model.forward(&train_images)?;
        // softmax the log probabilities
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        // get the loss
        let loss = loss::nll(&log_sm, &train_labels)?;
        // step the tensors by backpropagating the loss
        optimiser.backward_step(&loss)?;

        # // get the log probabilities of the test images
        # let test_logits = model.forward(&test_images)?;
        # // get the sum of the correct predictions
        # let sum_ok = test_logits
        #     .argmax(D::Minus1)?
        #     .eq(&test_labels)?
        #     .to_dtype(DType::F32)?
        #     .sum_all()?
        #     .to_scalar::<f32>()?;
        # // get the accuracy on the test set
        # #[allow(clippy::cast_precision_loss)]
        # let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        # println!(
        #     "{:4} train loss: {:8.5} test acc: {:5.2}%",
        #     epoch + 1,
        #     loss.to_scalar::<f32>()?,
        #     100. * test_accuracy
        # );
    }
    Ok(())
# }
```
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
pub trait LossOptimizer<'a, M: Model>: Sized {
    /// type of the optimiser configuration
    type Config: Sized;
    /// create a new optimiser from a Vec of variables, setup parameters and a model
    fn new(vs: Vec<Var>, params: Self::Config, model: &'a M) -> CResult<Self>;
    /// take a step of the optimiser
    fn backward_step(&mut self, loss: &Tensor) -> CResult<ModelOutcome>; //, xs: &Tensor, ys: &Tensor
    /// get the current learning rate
    fn learning_rate(&self) -> f64;
    /// set the learning rate
    fn set_learning_rate(&mut self, lr: f64);
    /// get the a vec of the variables being optimised
    fn into_inner(self) -> Vec<Var>;
    /// create a new optimiser from a slice of variables, setup parameters and a model
    fn from_slice(vars: &[&Var], config: Self::Config, model: &'a M) -> CResult<Self> {
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
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
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
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum Momentum {
    /// classical momentum
    Classical(f64),
    /// nesterov momentum
    Nesterov(f64),
}
