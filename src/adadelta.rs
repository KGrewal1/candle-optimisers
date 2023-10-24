use candle_core::{Result, Tensor, TensorId, Var};
use candle_nn::optim::Optimizer;
use std::collections::HashMap;

/// Adadelta optimizer
///
/// Described in <https://arxiv.org/abs/1212.5701>
///
/// For pseudocde see <https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html>

#[derive(Debug)]
pub struct Adadelta {
    vars: Vec<Var>,
    params: ParamsAdaDelta,
    avg_acc: HashMap<TensorId, (Tensor, Tensor)>,
}

#[derive(Debug)]
pub struct ParamsAdaDelta {
    pub lr: f64,
    pub rho: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for ParamsAdaDelta {
    fn default() -> Self {
        Self {
            lr: 1.0,
            rho: 0.9,
            weight_decay: 0.0,
            eps: 1e-6,
        }
    }
}

impl Optimizer for Adadelta {
    type Config = ParamsAdaDelta;

    fn new(vars: Vec<Var>, params: ParamsAdaDelta) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .collect();
        // // Err(SGDError::NoMomentum)?;
        // let mut params = params;
        // params.t = 0;
        Ok(Self {
            vars,
            params,
            avg_acc: HashMap::new(),
        })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        for var in &self.vars {
            if let Some(grad) = grads.get(var) {
                if self.params.weight_decay == 0. {
                    if let Some((v, u)) = self.avg_acc.get(&var.id()) {
                        let v =
                            ((v * self.params.rho)? + (1. - self.params.rho) * grad.powf(2.)?)?;
                        let delta_x = (((u + self.params.eps)?.powf(0.5)?)
                            .div(&((&v + self.params.eps)?.powf(0.5)?))?
                            * grad)?;
                        let u = ((u * self.params.rho)?
                            + (1. - self.params.rho) * delta_x.powf(2.)?)?;
                        var.set(&var.sub(&(delta_x * self.params.lr)?)?)?;
                        self.avg_acc.insert(var.id(), (v, u));
                    } else {
                        // start  u and v as 0 tensors
                        let v = ((1. - self.params.rho) * grad.powf(2.)?)?;
                        let delta_x = ((self.params.eps.powf(0.5))
                            * (&((&v + self.params.eps)?.powf(-0.5)?))
                            * grad)?;
                        let u = ((1. - self.params.rho) * delta_x.powf(2.)?)?;
                        var.set(&var.sub(&(delta_x * self.params.lr)?)?)?;
                        self.avg_acc.insert(var.id(), (v, u));
                    };
                } else {
                    let grad = &(grad + (self.params.weight_decay * var.as_tensor())?)?;
                    if let Some((v, u)) = self.avg_acc.get(&var.id()) {
                        let v =
                            ((v * self.params.rho)? + (1. - self.params.rho) * grad.powf(2.)?)?;
                        let delta_x = (((u + self.params.eps)?.powf(0.5)?)
                            .div(&((&v + self.params.eps)?.powf(0.5)?))?
                            * grad)?;
                        let u = ((u * self.params.rho)?
                            + (1. - self.params.rho) * delta_x.powf(2.)?)?;
                        var.set(&var.sub(&(delta_x * self.params.lr)?)?)?;
                        self.avg_acc.insert(var.id(), (v, u));
                    } else {
                        // start  u and v as 0 tensors
                        let v = ((1. - self.params.rho) * grad.powf(2.)?)?;
                        let delta_x = ((self.params.eps.powf(0.5))
                            * (&((&v + self.params.eps)?.powf(-0.5)?))
                            * grad)?;
                        let u = ((1. - self.params.rho) * delta_x.powf(2.)?)?;
                        var.set(&var.sub(&(delta_x * self.params.lr)?)?)?;
                        self.avg_acc.insert(var.id(), (v, u));
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

impl Adadelta {
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

        let params = ParamsAdaDelta {
            lr: 0.004,
            rho: 0.9,
            weight_decay: 0.0,
            eps: 1e-6,
        };
        // Now use backprop to run a linear regression between samples and get the coefficients back.
        let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
        let b = Var::new(0f32, &Device::Cpu)?;
        let mut n_sgd = Adadelta::new(vec![w.clone(), b.clone()], params)?;
        assert_eq!(0.004, n_sgd.learning_rate());
        n_sgd.set_learning_rate(0.002);
        assert_eq!(0.002, n_sgd.learning_rate());
        Ok(())
    }
}
