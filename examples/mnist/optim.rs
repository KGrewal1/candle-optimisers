use candle_core::{Result, Tensor, Var};
use candle_nn::Optimizer;
use optimisers::adadelta::{Adadelta, ParamsAdaDelta};
use optimisers::adagrad::{Adagrad, ParamsAdaGrad};
use optimisers::adam::{Adam, ParamsAdam};
use optimisers::adamax::{Adamax, ParamsAdaMax};
use optimisers::esgd::{ParamsSGD, SGD};
use optimisers::nadam::{NAdam, ParamsNAdam};
use optimisers::radam::{ParamsRAdam, RAdam};
use optimisers::rmsprop::{ParamsRMSprop, RMSprop};

pub trait Optim: Sized {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self>;
    fn back_step(&mut self, loss: &Tensor) -> Result<()>;
}

impl Optim for Adadelta {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        <Adadelta as Optimizer>::new(
            vars,
            ParamsAdaDelta {
                lr,
                ..Default::default()
            },
        )
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}

impl Optim for Adagrad {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        <Adagrad as Optimizer>::new(
            vars,
            ParamsAdaGrad {
                lr,
                ..Default::default()
            },
        )
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}

impl Optim for Adamax {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        <Adamax as Optimizer>::new(
            vars,
            ParamsAdaMax {
                lr,
                ..Default::default()
            },
        )
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}

impl Optim for SGD {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        <SGD as Optimizer>::new(
            vars,
            ParamsSGD {
                lr,
                ..Default::default()
            },
        )
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}

impl Optim for NAdam {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        <NAdam as Optimizer>::new(
            vars,
            ParamsNAdam {
                lr,
                ..Default::default()
            },
        )
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}

impl Optim for RAdam {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        <RAdam as Optimizer>::new(
            vars,
            ParamsRAdam {
                lr,
                ..Default::default()
            },
        )
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}

impl Optim for RMSprop {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        <RMSprop as Optimizer>::new(
            vars,
            ParamsRMSprop {
                lr,
                ..Default::default()
            },
        )
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}

impl Optim for Adam {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        <Adam as Optimizer>::new(
            vars,
            ParamsAdam {
                lr,
                ..Default::default()
            },
        )
    }

    fn back_step(&mut self, loss: &Tensor) -> Result<()> {
        self.backward_step(loss)
    }
}
