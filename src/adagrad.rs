use candle_core::{Result, Var};
use candle_nn::optim::Optimizer;

/// Adagrad optimiser
///
/// Described in <https://jmlr.org/papers/v12/duchi11a.html>
///
/// For pseudocde see <https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html>

#[derive(Debug)]
pub struct Adagrad {
    vars: Vec<VarAdaGrad>,
    params: ParamsAdaGrad,
    t: f64,
}

#[derive(Debug)]
struct VarAdaGrad {
    theta: Var,
    sum: Var,
}

#[derive(Debug)]
pub struct ParamsAdaGrad {
    pub lr: f64,
    pub lr_decay: f64,
    pub initial_acc: f64,
    pub dampening: f64,
    pub weight_decay: f64,
    pub eps: f64,
}

impl Default for ParamsAdaGrad {
    fn default() -> Self {
        Self {
            lr: 0.01,
            lr_decay: 0.0,
            initial_acc: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            eps: 1e-10,
        }
    }
}

impl Optimizer for Adagrad {
    type Config = ParamsAdaGrad;

    fn new(vars: Vec<Var>, params: ParamsAdaGrad) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let sum = Var::zeros(shape, dtype, device)?;
                Ok(VarAdaGrad { theta: var, sum })
            })
            .collect::<Result<Vec<VarAdaGrad>>>()?;
        // // Err(SGDError::NoMomentum)?;
        // let mut params = params;
        // params.t = 0;
        Ok(Self {
            vars,
            t: 0.,
            params,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        for var in &self.vars {
            let theta = &var.theta;
            let sum = &var.sum;
            if let Some(grad) = grads.get(theta) {
                let gamma_tilde = self.params.lr / (1. + (self.t * self.params.lr_decay));
                if self.params.weight_decay == 0. {
                    // let gt = (grad + (self.params.weight_decay * var.as_tensor())?)?;
                    let current_sum = (sum.as_tensor() + grad.powf(2.)?)?;
                    let change =
                        (gamma_tilde * (grad.div(&(current_sum.powf(0.5)? + self.params.eps)?))?)?;
                    sum.set(&current_sum)?;
                    theta.set(&theta.sub(&change)?)?;
                } else {
                    let grad = &(grad + (self.params.weight_decay * theta.as_tensor())?)?;
                    let current_sum = (sum.as_tensor() + grad.powf(2.)?)?;
                    let change =
                        (gamma_tilde * (grad.div(&(current_sum.powf(0.5)? + self.params.eps)?))?)?;
                    sum.set(&current_sum)?;
                    theta.set(&theta.sub(&change)?)?;
                }
            }
        }
        self.t += 1.;
        Ok(())
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }
}

impl Adagrad {
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

        let params = ParamsAdaGrad {
            lr: 0.004,
            ..Default::default()
        };
        // Now use backprop to run a linear regression between samples and get the coefficients back.
        let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
        let b = Var::new(0f32, &Device::Cpu)?;
        let mut n_sgd = Adagrad::new(vec![w.clone(), b.clone()], params)?;
        assert_approx_eq!(0.004, n_sgd.learning_rate());
        n_sgd.set_learning_rate(0.002);
        assert_approx_eq!(0.002, n_sgd.learning_rate());
        Ok(())
    }
}
