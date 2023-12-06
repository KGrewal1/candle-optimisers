//! Stochastic Gradient Descent with momentum, weight decay and Nestervov momentum

use candle_core::{Result, Var};
use candle_nn::optim::Optimizer;

use crate::Momentum;

/// Optimizer for Stochastic Gradient Descent with momentum.
///
/// Utilised same interface as pytorch but allows negative momenta and dampening with Nesterov
///
/// For pseudocde see <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>

#[derive(Debug)]
pub struct SGD {
    vars: Vec<VarSGD>,
    params: ParamsSGD,
}

#[derive(Debug)]
struct VarSGD {
    theta: Var,
    b: Option<Var>,
}

/// Parameters for SGD
#[derive(Debug)]
pub struct ParamsSGD {
    /// Learning rate
    pub lr: f64,
    /// Weight decay
    pub weight_decay: Option<f64>,
    /// Momentum
    pub momentum: Option<Momentum>,
    /// Dampening
    pub dampening: f64,
}

impl Default for ParamsSGD {
    fn default() -> Self {
        Self {
            lr: 0.1,
            weight_decay: None,
            momentum: None, //Momentum::Classical(0.1)
            dampening: 0.0,
            // nesterov: false,
        }
    }
}

impl Optimizer for SGD {
    type Config = ParamsSGD;

    fn new(vars: Vec<Var>, params: ParamsSGD) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| VarSGD {
                theta: var,
                b: None,
            })
            .collect::<Vec<VarSGD>>();
        // Err(SGDError::NoMomentum)?;
        Ok(Self { vars, params })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        if let Some(momentum) = self.params.momentum {
            match momentum {
                Momentum::Classical(momentum) => {
                    if let Some(wd) = self.params.weight_decay {
                        for var in &mut self.vars {
                            let theta = &var.theta;
                            // let prev_step = var.b;
                            if let Some(grad) = grads.get(theta) {
                                let grad = &(grad + (wd * theta.as_tensor())?)?;
                                if let Some(prev_step) = &(var.b) {
                                    // println!("Exists");
                                    // bt​←μbt−1​+(1−τ)gt
                                    let bt = ((prev_step.as_tensor() * momentum)?
                                        + (1. - self.params.dampening) * (grad))?;

                                    // if not nesterov gt = bt
                                    theta.set(&theta.sub(&(&bt * self.params.lr)?)?)?;
                                    // println!("Momentum {}", bt);
                                    prev_step.set(&bt)?;
                                } else {
                                    // println!("Doesn't Exist");
                                    // bt​←μbt−1​+(1−τ)gt
                                    // if there is no history bt = gt = grad with no weight_decay
                                    let bt = grad.clone(); // clone must occur invariably due to need to store in hashmap

                                    // if not nesterov gt = bt
                                    theta.set(&theta.sub(&(&bt * self.params.lr)?)?)?;
                                    // println!("Momentum {}", bt);
                                    var.b = Some(Var::from_tensor(&bt)?);
                                };
                            }
                        }
                    } else {
                        for var in &mut self.vars {
                            let theta = &var.theta;
                            // let prev_step = var.b;
                            if let Some(grad) = grads.get(theta) {
                                if let Some(prev_step) = &(var.b) {
                                    // println!("Exists");
                                    // bt​←μbt−1​+(1−τ)gt
                                    let bt = ((prev_step.as_tensor() * momentum)?
                                        + (1. - self.params.dampening) * (grad))?;

                                    // if not nesterov gt = bt
                                    theta.set(&theta.sub(&(&bt * self.params.lr)?)?)?;
                                    // println!("Momentum {}", bt);
                                    prev_step.set(&bt)?;
                                } else {
                                    // println!("Doesn't Exist");
                                    // bt​←μbt−1​+(1−τ)gt
                                    // if there is no history bt = gt = grad with no weight_decay
                                    let bt = grad.clone(); // clone must occur invariably due to need to store in hashmap

                                    // if not nesterov gt = bt
                                    theta.set(&theta.sub(&(&bt * self.params.lr)?)?)?;
                                    // println!("Momentum {}", bt);
                                    var.b = Some(Var::from_tensor(&bt)?);
                                };
                            }
                        }
                    }
                }
                Momentum::Nesterov(momentum) => {
                    if let Some(wd) = self.params.weight_decay {
                        for var in &mut self.vars {
                            let theta = &var.theta;
                            // let prev_step = var.b;
                            if let Some(grad) = grads.get(theta) {
                                let grad = &(grad + (wd * theta.as_tensor())?)?;
                                if let Some(prev_step) = &(var.b) {
                                    // println!("Exists");
                                    // bt​←μbt−1​+(1−τ)gt
                                    let bt = ((prev_step.as_tensor() * momentum)?
                                        + (1. - self.params.dampening) * (grad))?;

                                    let gt = (grad + (momentum * &bt)?)?;
                                    // println!("Momentum {}", bt);
                                    prev_step.set(&bt)?;
                                    theta.set(&theta.sub(&(gt * self.params.lr)?)?)?;
                                } else {
                                    // println!("Doesn't Exist");
                                    // bt​←μbt−1​+(1−τ)gt
                                    // if there is no history bt = gt = grad with no weight_decay
                                    let bt = grad.clone(); // clone must occur invariably due to need to store in hashmap

                                    let gt = (grad + (momentum * &bt)?)?;
                                    // println!("Momentum {}", bt);
                                    var.b = Some(Var::from_tensor(&bt)?);
                                    theta.set(&theta.sub(&(gt * self.params.lr)?)?)?;
                                };
                            }
                        }
                    } else {
                        for var in &mut self.vars {
                            let theta = &var.theta;
                            // let prev_step = var.b;
                            if let Some(grad) = grads.get(theta) {
                                if let Some(prev_step) = &(var.b) {
                                    // println!("Exists");
                                    // bt​←μbt−1​+(1−τ)gt
                                    let bt = ((prev_step.as_tensor() * momentum)?
                                        + (1. - self.params.dampening) * (grad))?;

                                    let gt = (grad + (momentum * &bt)?)?;
                                    // println!("Momentum {}", bt);
                                    prev_step.set(&bt)?;
                                    theta.set(&theta.sub(&(gt * self.params.lr)?)?)?;
                                } else {
                                    // println!("Doesn't Exist");
                                    // bt​←μbt−1​+(1−τ)gt
                                    // if there is no history bt = gt = grad with no weight_decay
                                    let bt = grad.clone(); // clone must occur invariably due to need to store in hashmap

                                    let gt = (grad + (momentum * &bt)?)?;
                                    // println!("Momentum {}", bt);
                                    var.b = Some(Var::from_tensor(&bt)?);
                                    theta.set(&theta.sub(&(gt * self.params.lr)?)?)?;
                                };
                            }
                        }
                    }
                }
            }
        } else if let Some(wd) = self.params.weight_decay {
            for var in &mut self.vars {
                let theta = &var.theta;
                // let prev_step = var.b;
                if let Some(grad) = grads.get(theta) {
                    let grad = &(grad + (wd * theta.as_tensor())?)?; // weight decay grad
                    theta.set(&theta.sub(&(grad * self.params.lr)?)?)?; // update theta
                }
            }
        } else {
            for var in &mut self.vars {
                let theta = &var.theta;
                // let prev_step = var.b;
                if let Some(grad) = grads.get(theta) {
                    theta.set(&theta.sub(&(grad * self.params.lr)?)?)?; // update theta based on grad
                }
            }
        }

        Ok(())
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }
}

impl SGD {
    #[must_use]
    pub fn into_inner(self) -> Vec<Var> {
        self.vars.into_iter().map(|v| v.theta).collect()
    }

    // pub fn push(&mut self, var: &Var) {
    //     self.vars.push(var.clone());
    // }
}

#[cfg(test)]
mod tests {
    // use candle_core::test_utils::{to_vec0_round, to_vec2_round};

    use anyhow::Result;
    use assert_approx_eq::assert_approx_eq;
    use candle_core::{Device, Var};
    use candle_nn::Optimizer;

    use super::*;
    #[test]
    fn lr_test() -> Result<()> {
        let params = ParamsSGD {
            lr: 0.004,
            ..Default::default()
        };
        // Now use backprop to run a linear regression between samples and get the coefficients back.
        let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
        let b = Var::new(0f32, &Device::Cpu)?;
        let mut n_sgd = SGD::new(vec![w.clone(), b.clone()], params)?;
        assert_approx_eq!(0.004, n_sgd.learning_rate());
        n_sgd.set_learning_rate(0.002);
        assert_approx_eq!(0.002, n_sgd.learning_rate());
        Ok(())
    }

    #[test]
    fn into_inner_test() -> Result<()> {
        let params = ParamsSGD::default();
        let w = Var::new(&[[3f32, 1.]], &Device::Cpu)?;
        let b = Var::new(-2f32, &Device::Cpu)?;
        let n_sgd = SGD::new(vec![w.clone(), b.clone()], params)?;
        let inner = n_sgd.into_inner();
        assert_eq!(inner[0].as_tensor().to_vec2::<f32>()?, &[[3f32, 1.]]);
        assert_approx_eq!(inner[1].as_tensor().to_vec0::<f32>()?, -2_f32);
        Ok(())
    }
}
