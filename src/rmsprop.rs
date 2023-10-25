use candle_core::{Result, Tensor, TensorId, Var};
use candle_nn::optim::Optimizer;
use std::collections::HashMap;

/// R Adam optimizer
///
/// Described in <https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ>
///
/// For pseudocde see <https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam>

#[derive(Debug)]
pub struct RMSprop {
    vars: Vec<Var>,
    params: ParamsRMSprop,
    v_buffer_gavg: HashMap<TensorId, (Tensor, Option<Tensor>, Option<Tensor>)>,
}

#[derive(Debug)]
pub struct ParamsRMSprop {
    pub lr: f64,
    pub alpha: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub momentum: f64,
    pub centered: bool,
}

impl Default for ParamsRMSprop {
    fn default() -> Self {
        Self {
            lr: 0.01,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.,
            momentum: 0.,
            centered: false,
        }
    }
}

impl Optimizer for RMSprop {
    type Config = ParamsRMSprop;

    fn new(vars: Vec<Var>, params: ParamsRMSprop) -> Result<Self> {
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
            v_buffer_gavg: HashMap::new(),
        })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        // println!("prod {}", prod);

        for var in &self.vars {
            if let Some(grad) = grads.get(var) {
                if self.params.weight_decay == 0. {
                    if let Some((v, b, g_avg)) = self.v_buffer_gavg.get(&var.id()) {
                        let v = ((self.params.alpha * v)?
                            + ((1. - self.params.alpha) * grad.powf(2.)?)?)?;
                        let (v_tilde, g_avg) = if let Some(g) = g_avg {
                            let g =
                                ((self.params.alpha * g)? + ((1. - self.params.alpha) * grad)?)?;
                            ((&v - g.powf(2.)?)?, Some(g))
                        } else {
                            (v.clone(), None)
                        };
                        if let Some(b) = b {
                            let b = ((self.params.momentum * b)?
                                + (grad / (v_tilde.sqrt()? + self.params.eps)?)?)?;
                            var.set(&var.sub(&(self.params.lr * &b)?)?)?;
                            self.v_buffer_gavg.insert(var.id(), (v, Some(b), g_avg));
                        } else {
                            let delta = (grad / (v_tilde.sqrt()? + self.params.eps)?)?;
                            var.set(&var.sub(&(self.params.lr * delta)?)?)?;
                            self.v_buffer_gavg.insert(var.id(), (v, None, g_avg));
                        }
                    } else {
                        let v = ((1. - self.params.alpha) * grad.powf(2.)?)?;
                        let (v_tilde, g_avg) = if self.params.centered {
                            let g = ((1. - self.params.alpha) * grad)?;
                            ((&v - g.powf(2.)?)?, Some(g))
                        } else {
                            (v.clone(), None)
                        };
                        if self.params.momentum == 0. {
                            let delta = (grad / (v_tilde.sqrt()? + self.params.eps)?)?;
                            var.set(&var.sub(&(self.params.lr * delta)?)?)?;
                            self.v_buffer_gavg.insert(var.id(), (v, None, g_avg));
                        } else {
                            let b = (grad / (v_tilde.sqrt()? + self.params.eps)?)?;
                            var.set(&var.sub(&(self.params.lr * &b)?)?)?;
                            self.v_buffer_gavg.insert(var.id(), (v, Some(b), g_avg));
                        }
                    };
                } else {
                    let grad = &(grad + (self.params.weight_decay * var.as_tensor())?)?;
                    if let Some((v, b, g_avg)) = self.v_buffer_gavg.get(&var.id()) {
                        let v = ((self.params.alpha * v)?
                            + ((1. - self.params.alpha) * grad.powf(2.)?)?)?;
                        let (v_tilde, g_avg) = if let Some(g) = g_avg {
                            let g =
                                ((self.params.alpha * g)? + ((1. - self.params.alpha) * grad)?)?;
                            ((&v - g.powf(2.)?)?, Some(g))
                        } else {
                            (v.clone(), None)
                        };
                        if let Some(b) = b {
                            let b = ((self.params.momentum * b)?
                                + (grad / (v_tilde.sqrt()? + self.params.eps)?)?)?;
                            var.set(&var.sub(&(self.params.lr * &b)?)?)?;
                            self.v_buffer_gavg.insert(var.id(), (v, Some(b), g_avg));
                        } else {
                            let delta = (grad / (v_tilde.sqrt()? + self.params.eps)?)?;
                            var.set(&var.sub(&(self.params.lr * delta)?)?)?;
                            self.v_buffer_gavg.insert(var.id(), (v, None, g_avg));
                        }
                    } else {
                        let v = ((1. - self.params.alpha) * grad.powf(2.)?)?;
                        let (v_tilde, g_avg) = if self.params.centered {
                            let g = ((1. - self.params.alpha) * grad)?;
                            ((&v - g.powf(2.)?)?, Some(g))
                        } else {
                            (v.clone(), None)
                        };
                        if self.params.momentum == 0. {
                            let delta = (grad / (v_tilde.sqrt()? + self.params.eps)?)?;
                            var.set(&var.sub(&(self.params.lr * delta)?)?)?;
                            self.v_buffer_gavg.insert(var.id(), (v, None, g_avg));
                        } else {
                            let b = (grad / (v_tilde.sqrt()? + self.params.eps)?)?;
                            var.set(&var.sub(&(self.params.lr * &b)?)?)?;
                            self.v_buffer_gavg.insert(var.id(), (v, Some(b), g_avg));
                        }
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

impl RMSprop {
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
    use assert_approx_eq::assert_approx_eq;
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

        let params = ParamsRMSprop {
            lr: 0.004,
            ..Default::default()
        };
        // Now use backprop to run a linear regression between samples and get the coefficients back.
        let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
        let b = Var::new(0f32, &Device::Cpu)?;
        let mut n_sgd = RMSprop::new(vec![w.clone(), b.clone()], params)?;
        assert_approx_eq!(0.004, n_sgd.learning_rate());
        n_sgd.set_learning_rate(0.002);
        assert_approx_eq!(0.002, n_sgd.learning_rate());
        Ok(())
    }
}
