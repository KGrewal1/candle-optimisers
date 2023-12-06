use candle_core::{DType, Result, Tensor, Var, D};
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap};

use optimisers::adadelta::{Adadelta, ParamsAdaDelta};
use optimisers::adagrad::{Adagrad, ParamsAdaGrad};
use optimisers::adam::{Adam, ParamsAdam};
use optimisers::adamax::{Adamax, ParamsAdaMax};
use optimisers::esgd::{MomentumEnhancedSGD, ParamsMESGD};
use optimisers::lbfgs::{Lbfgs, LineSearch, ParamsLBFGS};
use optimisers::nadam::{NAdam, ParamsNAdam};
use optimisers::radam::{ParamsRAdam, RAdam};
use optimisers::rmsprop::{ParamsRMSprop, RMSprop};
use optimisers::{LossOptimizer, Model};

pub trait Optim: Sized {
    fn new(vars: Vec<Var>) -> Result<Self>;
    fn back_step(&mut self, loss: &Tensor) -> Result<()>;
}

impl Optim for Adadelta {
    fn new(vars: Vec<Var>) -> Result<Self> {
        <Adadelta as Optimizer>::new(vars, ParamsAdaDelta::default())
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}

impl Optim for Adagrad {
    fn new(vars: Vec<Var>) -> Result<Self> {
        <Adagrad as Optimizer>::new(vars, ParamsAdaGrad::default())
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}

impl Optim for Adamax {
    fn new(vars: Vec<Var>) -> Result<Self> {
        <Adamax as Optimizer>::new(vars, ParamsAdaMax::default())
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}

impl Optim for MomentumEnhancedSGD {
    fn new(vars: Vec<Var>) -> Result<Self> {
        <MomentumEnhancedSGD as Optimizer>::new(vars, ParamsMESGD::default())
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}

impl Optim for NAdam {
    fn new(vars: Vec<Var>) -> Result<Self> {
        <NAdam as Optimizer>::new(vars, ParamsNAdam::default())
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}

impl Optim for RAdam {
    fn new(vars: Vec<Var>) -> Result<Self> {
        <RAdam as Optimizer>::new(vars, ParamsRAdam::default())
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}

impl Optim for RMSprop {
    fn new(vars: Vec<Var>) -> Result<Self> {
        <RMSprop as Optimizer>::new(vars, ParamsRMSprop::default())
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}

impl Optim for Adam {
    fn new(vars: Vec<Var>) -> Result<Self> {
        <Adam as Optimizer>::new(vars, ParamsAdam::default())
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}

const IMAGE_DIM: usize = 784;
const LABELS: usize = 10;

pub trait SimpleModel: Sized {
    fn new(vs: VarBuilder, train_data: Tensor, train_labels: Tensor) -> Result<Self>;
    fn forward(&self) -> Result<Tensor>;
}

pub struct Mlp {
    ln1: Linear,
    ln2: Linear,
    train_data: Tensor,
    train_labels: Tensor,
}

impl SimpleModel for Mlp {
    fn new(vs: VarBuilder, train_data: Tensor, train_labels: Tensor) -> Result<Self> {
        let ln1 = candle_nn::linear(IMAGE_DIM, 100, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(100, LABELS, vs.pp("ln2"))?;
        Ok(Self {
            ln1,
            ln2,
            train_data,
            train_labels,
        })
    }

    fn forward(&self) -> Result<Tensor> {
        let xs = self.ln1.forward(&self.train_data)?;
        let xs = xs.relu()?;
        self.ln2.forward(&xs)
    }
}

impl Model for Mlp {
    fn loss(&self) -> Result<Tensor> {
        let logits = self.forward()?;
        // softmax the log probabilities
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        // get the loss
        loss::nll(&log_sm, &self.train_labels)
    }
}

#[allow(clippy::module_name_repetitions)]
pub fn run_training<M: SimpleModel + Model, O: Optim>(
    m: &candle_datasets::vision::Dataset,
) -> anyhow::Result<()> {
    // check to see if cuda device availabke
    let dev = candle_core::Device::cuda_if_available(0)?;

    // get the labels from the dataset
    let train_labels = m.train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    // get the input from the dataset and put on device
    let train_images = m.train_images.to_device(&dev)?;

    // create a new variable store
    let varmap = VarMap::new();
    // create a new variable builder
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    // create model from variables
    let model = M::new(vs.clone(), train_images, train_labels)?;

    // create an optimiser
    let mut optimiser = O::new(varmap.all_vars())?;
    // load the test images
    let _test_images = m.test_images.to_device(&dev)?;
    // load the test labels
    let _test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    for _epoch in 0..100 {
        // get the loss
        let loss = model.loss()?;
        // step the tensors by backpropagating the loss
        optimiser.back_step(&loss)?;
    }
    Ok(())
}

pub fn run_lbfgs_training<M: SimpleModel + Model>(
    m: &candle_datasets::vision::Dataset,
) -> anyhow::Result<()> {
    // check to see if cuda device availabke
    let dev = candle_core::Device::cuda_if_available(0)?;

    // get the labels from the dataset
    let train_labels = m.train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    // get the input from the dataset and put on device
    let train_images = m.train_images.to_device(&dev)?;

    // create a new variable store
    let varmap = VarMap::new();
    // create a new variable builder
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    // create model from variables
    let model = M::new(vs.clone(), train_images, train_labels)?;

    let params = ParamsLBFGS {
        lr: 1.,
        history_size: 4,
        line_search: Some(LineSearch::StrongWolfe(1e-4, 0.9, 1e-9)),
        ..Default::default()
    };

    let mut loss = model.loss()?;

    // create an optimiser
    let mut optimiser = Lbfgs::new(varmap.all_vars(), params, model)?;
    // load the test images
    let _test_images = m.test_images.to_device(&dev)?;
    // load the test labels
    let _test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    for _epoch in 0..100 {
        // get the loss

        // step the tensors by backpropagating the loss
        let res = optimiser.backward_step(&loss)?;
        match res {
            optimisers::ModelOutcome::Converged(_, _) => break,
            optimisers::ModelOutcome::Stepped(new_loss, _) => loss = new_loss,
            // _ => panic!("unexpected outcome"),
        }
    }
    Ok(())
}
