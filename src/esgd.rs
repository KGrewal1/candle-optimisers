use candle_core::{Result, Tensor, TensorId, Var};
use candle_nn::optim::Optimizer;
use std::collections::HashMap;

/// Optimizer for Stochastic Gradient Descent with momentum.
///
/// Utilised same interface as pytorch but allows negative momenta and dampening with Nesterov
///
/// For pseudocde see <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>

#[derive(Debug)]
pub struct MomentumEnhancedSGD {
    vars: Vec<Var>,
    params: ParamsMESGD,
    prev_step: HashMap<TensorId, Tensor>,
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
            .collect();
        // Err(SGDError::NoMomentum)?;
        Ok(Self {
            vars,
            params,
            prev_step: HashMap::new(),
        })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        for var in &self.vars {
            if let Some(grad) = grads.get(var) {
                if self.params.weight_decay == 0. {
                    if let Some(prev_step) = self.prev_step.get(&var.id()) {
                        // println!("Exists");
                        // bt​←μbt−1​+(1−τ)gt
                        let bt = ((prev_step * self.params.momentum)?
                            + (1. - self.params.dampening) * (grad))?;
                        if self.params.nesterov {
                            let gt = (grad + (self.params.momentum * &bt)?)?;
                            // println!("Momentum {}", bt);
                            self.prev_step.insert(var.id(), bt);
                            var.set(&var.sub(&(gt * self.params.lr)?)?)?;
                        } else {
                            // if not nesterov gt = bt
                            var.set(&var.sub(&(&bt * self.params.lr)?)?)?;
                            // println!("Momentum {}", bt);
                            self.prev_step.insert(var.id(), bt);
                        };
                    } else {
                        // println!("Doesn't Exist");
                        // bt​←μbt−1​+(1−τ)gt
                        // if there is no history bt = gt = grad with no weight_decay
                        let bt = grad.clone(); // clone must occur invariably due to need to store in hashmap
                        if self.params.nesterov {
                            let gt = (grad + (self.params.momentum * &bt)?)?;
                            // println!("Momentum {}", bt);
                            self.prev_step.insert(var.id(), bt);
                            var.set(&var.sub(&(gt * self.params.lr)?)?)?;
                        } else {
                            // if not nesterov gt = bt
                            var.set(&var.sub(&(&bt * self.params.lr)?)?)?;
                            // println!("Momentum {}", bt);
                            self.prev_step.insert(var.id(), bt);
                        };
                    };
                } else {
                    let grad = &(grad + (self.params.weight_decay * var.as_tensor())?)?;
                    if let Some(prev_step) = self.prev_step.get(&var.id()) {
                        // println!("Exists");
                        // bt​←μbt−1​+(1−τ)gt
                        let bt = ((prev_step * self.params.momentum)?
                            + (1. - self.params.dampening) * (grad))?;
                        if self.params.nesterov {
                            let gt = (grad + (self.params.momentum * &bt)?)?;
                            // println!("Momentum {}", bt);
                            self.prev_step.insert(var.id(), bt);
                            var.set(&var.sub(&(gt * self.params.lr)?)?)?;
                        } else {
                            // if not nesterov gt = bt
                            var.set(&var.sub(&(&bt * self.params.lr)?)?)?;
                            // println!("Momentum {}", bt);
                            self.prev_step.insert(var.id(), bt);
                        };
                    } else {
                        // println!("Doesn't Exist");
                        // bt​←μbt−1​+(1−τ)gt
                        // if there is no history bt = gt = grad with no weight_decay
                        let bt = grad.clone(); // clone must occur invariably due to need to store in hashmap
                        if self.params.nesterov {
                            let gt = (grad + (self.params.momentum * &bt)?)?;
                            // println!("Momentum {}", bt);
                            self.prev_step.insert(var.id(), bt);
                            var.set(&var.sub(&(gt * self.params.lr)?)?)?;
                        } else {
                            // if not nesterov gt = bt
                            var.set(&var.sub(&(&bt * self.params.lr)?)?)?;
                            // println!("Momentum {}", bt);
                            self.prev_step.insert(var.id(), bt);
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
        self.vars
    }

    pub fn push(&mut self, var: &Var) {
        self.vars.push(var.clone());
    }
}

#[cfg(test)]
mod tests {
    // use candle_core::test_utils::{to_vec0_round, to_vec2_round};

    use anyhow::Result;
    use candle_core::{Device, Tensor, Var};
    use candle_nn::{Linear, Module, Optimizer};

    use super::*;
    #[test]
    fn insertiontest() -> Result<()> {
        let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
        let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
        let gen = Linear::new(w_gen, Some(b_gen));
        let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
        let _sample_ys = gen.forward(&sample_xs)?;

        let params = ParamsMESGD {
            lr: 0.004,
            momentum: 0.1,
            weight_decay: 0.0,
            dampening: 0.2,
            nesterov: false,
        };
        // Now use backprop to run a linear regression between samples and get the coefficients back.
        let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
        let b = Var::new(0f32, &Device::Cpu)?;
        let mut n_sgd = MomentumEnhancedSGD::new(vec![w.clone(), b.clone()], params)?;
        assert_eq!(0.004, n_sgd.learning_rate());
        n_sgd.set_learning_rate(0.002);
        assert_eq!(0.002, n_sgd.learning_rate());
        Ok(())
    }
}
