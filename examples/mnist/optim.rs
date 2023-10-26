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
    fn backward_step(&mut self, loss: &Tensor) -> Result<()>;
}

pub struct AdaDelta {
    adadelta: Adadelta,
}

impl Optim for AdaDelta {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        let adadelta = Adadelta::new(
            vars,
            ParamsAdaDelta {
                lr,
                ..Default::default()
            },
        )?;
        Ok(Self { adadelta })
    }

    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        self.adadelta.backward_step(loss)
    }
}

pub struct AdaGrad {
    adagrad: Adagrad,
}

impl Optim for AdaGrad {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        let adagrad = Adagrad::new(
            vars,
            ParamsAdaGrad {
                lr,
                ..Default::default()
            },
        )?;
        Ok(Self { adagrad })
    }

    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        self.adagrad.backward_step(loss)
    }
}

pub struct AdaMax {
    adamax: Adamax,
}

impl Optim for AdaMax {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        let adamax = Adamax::new(
            vars,
            ParamsAdaMax {
                lr,
                ..Default::default()
            },
        )?;
        Ok(Self { adamax })
    }

    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        self.adamax.backward_step(loss)
    }
}

pub struct SGD {
    sgd: MomentumEnhancedSGD,
}

impl Optim for SGD {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        let sgd = MomentumEnhancedSGD::new(
            vars,
            ParamsMESGD {
                lr,
                ..Default::default()
            },
        )?;
        Ok(Self { sgd })
    }

    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        self.sgd.backward_step(loss)
    }
}

pub struct NsAdam {
    nadam: NAdam,
}

impl Optim for NsAdam {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        let nadam = NAdam::new(
            vars,
            ParamsNAdam {
                lr,
                ..Default::default()
            },
        )?;
        Ok(Self { nadam })
    }

    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        self.nadam.backward_step(loss)
    }
}

pub struct RsAdam {
    radam: RAdam,
}

impl Optim for RsAdam {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        let radam = RAdam::new(
            vars,
            ParamsRAdam {
                lr,
                ..Default::default()
            },
        )?;
        Ok(Self { radam })
    }

    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        self.radam.backward_step(loss)
    }
}

pub struct RMS {
    rms: RMSprop,
}

impl Optim for RMS {
    fn new(vars: Vec<Var>, lr: f64) -> Result<Self> {
        let rms = RMSprop::new(
            vars,
            ParamsRMSprop {
                lr,
                ..Default::default()
            },
        )?;
        Ok(Self { rms })
    }

    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        self.rms.backward_step(loss)
    }
}
