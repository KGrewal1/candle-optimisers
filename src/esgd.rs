//! Stochastic Gradient Descent with momentum, weight decay and Nestervov momentum

use candle_core::{Result, Var};
use candle_nn::optim::Optimizer;

/// Optimizer for Stochastic Gradient Descent with momentum.
///
/// Utilised same interface as pytorch but allows negative momenta and dampening with Nesterov
///
/// For pseudocde see <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>

#[derive(Debug)]
pub struct MomentumEnhancedSGD {
    vars: Vec<VarMESGD>,
    params: ParamsMESGD,
}

#[derive(Debug)]
struct VarMESGD {
    theta: Var,
    b: Option<Var>,
}

#[derive(Debug)]
pub struct ParamsMESGD {
    pub lr: f64,
    pub weight_decay: f64,
    pub momentum: f64,
    pub dampening: f64,
    pub nesterov: bool,
}

impl Default for ParamsMESGD {
    fn default() -> Self {
        Self {
            lr: 0.1,
            weight_decay: 0.,
            momentum: 0.1,
            dampening: 0.0,
            nesterov: false,
        }
    }
}

impl Optimizer for MomentumEnhancedSGD {
    type Config = ParamsMESGD;

    fn new(vars: Vec<Var>, params: ParamsMESGD) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| VarMESGD {
                theta: var,
                b: None,
            })
            .collect::<Vec<VarMESGD>>();
        // Err(SGDError::NoMomentum)?;
        Ok(Self { vars, params })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        for var in &mut self.vars {
            let theta = &var.theta;
            // let prev_step = var.b;
            if let Some(grad) = grads.get(theta) {
                if self.params.weight_decay == 0. {
                    if let Some(prev_step) = &(var.b) {
                        // println!("Exists");
                        // bt​←μbt−1​+(1−τ)gt
                        let bt = ((prev_step.as_tensor() * self.params.momentum)?
                            + (1. - self.params.dampening) * (grad))?;
                        if self.params.nesterov {
                            let gt = (grad + (self.params.momentum * &bt)?)?;
                            // println!("Momentum {}", bt);
                            prev_step.set(&bt)?;
                            theta.set(&theta.sub(&(gt * self.params.lr)?)?)?;
                        } else {
                            // if not nesterov gt = bt
                            theta.set(&theta.sub(&(&bt * self.params.lr)?)?)?;
                            // println!("Momentum {}", bt);
                            prev_step.set(&bt)?;
                        };
                    } else {
                        // println!("Doesn't Exist");
                        // bt​←μbt−1​+(1−τ)gt
                        // if there is no history bt = gt = grad with no weight_decay
                        let bt = grad.clone(); // clone must occur invariably due to need to store in hashmap
                        if self.params.nesterov {
                            let gt = (grad + (self.params.momentum * &bt)?)?;
                            // println!("Momentum {}", bt);
                            var.b = Some(Var::from_tensor(&bt)?);
                            theta.set(&theta.sub(&(gt * self.params.lr)?)?)?;
                        } else {
                            // if not nesterov gt = bt
                            theta.set(&theta.sub(&(&bt * self.params.lr)?)?)?;
                            // println!("Momentum {}", bt);
                            var.b = Some(Var::from_tensor(&bt)?);
                        };
                    };
                } else {
                    let grad = &(grad + (self.params.weight_decay * theta.as_tensor())?)?;
                    if let Some(prev_step) = &(var.b) {
                        // println!("Exists");
                        // bt​←μbt−1​+(1−τ)gt
                        let bt = ((prev_step.as_tensor() * self.params.momentum)?
                            + (1. - self.params.dampening) * (grad))?;
                        if self.params.nesterov {
                            let gt = (grad + (self.params.momentum * &bt)?)?;
                            // println!("Momentum {}", bt);
                            prev_step.set(&bt)?;
                            theta.set(&theta.sub(&(gt * self.params.lr)?)?)?;
                        } else {
                            // if not nesterov gt = bt
                            theta.set(&theta.sub(&(&bt * self.params.lr)?)?)?;
                            // println!("Momentum {}", bt);
                            prev_step.set(&bt)?;
                        };
                    } else {
                        // println!("Doesn't Exist");
                        // bt​←μbt−1​+(1−τ)gt
                        // if there is no history bt = gt = grad with no weight_decay
                        let bt = grad.clone(); // clone must occur invariably due to need to store in hashmap
                        if self.params.nesterov {
                            let gt = (grad + (self.params.momentum * &bt)?)?;
                            // println!("Momentum {}", bt);
                            var.b = Some(Var::from_tensor(&bt)?);
                            theta.set(&theta.sub(&(gt * self.params.lr)?)?)?;
                        } else {
                            // if not nesterov gt = bt
                            theta.set(&theta.sub(&(&bt * self.params.lr)?)?)?;
                            // println!("Momentum {}", bt);
                            var.b = Some(Var::from_tensor(&bt)?);
                        };
                    };
                }
            }
        }
        Ok(())
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }
}

impl MomentumEnhancedSGD {
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
        let params = ParamsMESGD {
            lr: 0.004,
            ..Default::default()
        };
        // Now use backprop to run a linear regression between samples and get the coefficients back.
        let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
        let b = Var::new(0f32, &Device::Cpu)?;
        let mut n_sgd = MomentumEnhancedSGD::new(vec![w.clone(), b.clone()], params)?;
        assert_approx_eq!(0.004, n_sgd.learning_rate());
        n_sgd.set_learning_rate(0.002);
        assert_approx_eq!(0.002, n_sgd.learning_rate());
        Ok(())
    }

    #[test]
    fn into_inner_test() -> Result<()> {
        let params = ParamsMESGD::default();
        let w = Var::new(&[[3f32, 1.]], &Device::Cpu)?;
        let b = Var::new(-2f32, &Device::Cpu)?;
        let n_sgd = MomentumEnhancedSGD::new(vec![w.clone(), b.clone()], params)?;
        let inner = n_sgd.into_inner();
        assert_eq!(inner[0].as_tensor().to_vec2::<f32>()?, &[[3f32, 1.]]);
        assert_approx_eq!(inner[1].as_tensor().to_vec0::<f32>()?, -2_f32);
        Ok(())
    }
}
