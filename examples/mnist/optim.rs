use candle_core::{Result, Tensor, Var};
use candle_nn::Optimizer;
use optimisers::adadelta::{Adadelta, ParamsAdaDelta};
use optimisers::adagrad::{Adagrad, ParamsAdaGrad};
use optimisers::adamax::{Adamax, ParamsAdaMax};
use optimisers::esgd::{MomentumEnhancedSGD, ParamsMESGD};
use optimisers::nadam::{NAdam, ParamsNAdam};
use optimisers::radam::{ParamsRAdam, RAdam};
use optimisers::rmsprop::{ParamsRMSprop, RMSprop};

pub trait Optim: Sized {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self>;
    fn back_step(&mut self, loss: &Tensor) -> Result<()>;
}

impl Optim for Adadelta {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        Ok(<Adadelta as Optimizer>::new(
            vars,
            ParamsAdaDelta {
                lr,
                ..Default::default()
            },
        )?)
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}

impl Optim for Adagrad {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        Ok(<Adagrad as Optimizer>::new(
            vars,
            ParamsAdaGrad {
                lr,
                ..Default::default()
            },
        )?)
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}

impl Optim for Adamax {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        Ok(<Adamax as Optimizer>::new(
            vars,
            ParamsAdaMax {
                lr,
                ..Default::default()
            },
        )?)
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}

impl Optim for MomentumEnhancedSGD {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        Ok(<MomentumEnhancedSGD as Optimizer>::new(
            vars,
            ParamsMESGD {
                lr,
                ..Default::default()
            },
        )?)
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}

impl Optim for NAdam {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        Ok(<NAdam as Optimizer>::new(
            vars,
            ParamsNAdam {
                lr,
                ..Default::default()
            },
        )?)
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}

impl Optim for RAdam {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        Ok(<RAdam as Optimizer>::new(
            vars,
            ParamsRAdam {
                lr,
                ..Default::default()
            },
        )?)
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}

impl Optim for RMSprop {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        Ok(<RMSprop as Optimizer>::new(
            vars,
            ParamsRMSprop {
                lr,
                ..Default::default()
            },
        )?)
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}
