//! The R Adam optimiser
//!
//! Described in [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265)
//!
//! For pseudocde see <https://pytorch.org/docs/stable/generated/torch.optim.RAdam.html>

use candle_core::{Result, Var};
use candle_nn::optim::Optimizer;

/// R Adam optimiser
///
/// Described in [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265)
///
/// For pseudocde see <https://pytorch.org/docs/stable/generated/torch.optim.RAdam.html>

#[derive(Debug)]
pub struct RAdam {
    vars: Vec<VarRAdam>,
    params: ParamsRAdam,
    rho_inf: f64,
    t: f64,
}

#[derive(Debug)]
struct VarRAdam {
    theta: Var,
    m: Var,
    v: Var,
}

/// Parameters for the R Adam optimiser
#[derive(Debug)]
pub struct ParamsRAdam {
    /// Learning rate
    pub lr: f64,
    /// Coefficient for moving average of first moment
    pub beta_1: f64,
    /// Coefficient for moving average of second moment
    pub beta_2: f64,
    /// Weight decay
    pub weight_decay: Option<f64>,
    /// Term added to denominator to improve numerical stability
    pub eps: f64,
}

impl Default for ParamsRAdam {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta_1: 0.9,
            beta_2: 0.999,
            eps: 1e-8,
            weight_decay: None,
        }
    }
}

impl Optimizer for RAdam {
    type Config = ParamsRAdam;

    fn new(vars: Vec<Var>, params: ParamsRAdam) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let m = Var::zeros(shape, dtype, device)?;
                let v = Var::zeros(shape, dtype, device)?;
                Ok(VarRAdam { theta: var, m, v })
            })
            .collect::<Result<Vec<VarRAdam>>>()?;
        // // Err(SGDError::NoMomentum)?;
        // let mut params = params;
        // params.t = 0;
        let rho_inf = 2. / (1. - params.beta_2) - 1.;
        Ok(Self {
            vars,
            params,
            rho_inf,
            t: 1.,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        // println!("prod {}", prod);
        let rho_t = self.rho_inf
            - 2. * self.t * self.params.beta_2.powf(self.t)
                / (1. - self.params.beta_2.powf(self.t));

        if let Some(wd) = self.params.weight_decay {
            for var in &self.vars {
                let theta = &var.theta;
                let m = &var.m;
                let v = &var.v;
                if let Some(grad) = grads.get(theta) {
                    let grad = &(grad + (wd * theta.as_tensor())?)?;
                    let m_next = ((self.params.beta_1 * m.as_tensor())?
                        + ((1. - self.params.beta_1) * grad)?)?;
                    let v_next = ((self.params.beta_2 * v.as_tensor())?
                        + ((1. - self.params.beta_2) * grad.powf(2.)?)?)?;
                    let m_hat = (&m_next / (1. - self.params.beta_1.powf(self.t)))?;

                    let delta = if rho_t > 5. {
                        let l = ((1. - self.params.beta_2.powf(self.t)).sqrt()
                            / (&v_next.sqrt()? + self.params.eps)?)?;
                        let r = ((rho_t - 4.) * (rho_t - 2.) * self.rho_inf
                            / ((self.rho_inf - 4.) * (self.rho_inf - 2.) * rho_t))
                            .sqrt();
                        (self.params.lr * r * (l * m_hat)?)?
                    } else {
                        (self.params.lr * m_hat)?
                    };
                    theta.set(&theta.sub(&(delta))?)?;
                    m.set(&m_next)?;
                    v.set(&v_next)?;
                }
            }
        } else {
            for var in &self.vars {
                let theta = &var.theta;
                let m = &var.m;
                let v = &var.v;
                if let Some(grad) = grads.get(theta) {
                    let m_next = ((self.params.beta_1 * m.as_tensor())?
                        + ((1. - self.params.beta_1) * grad)?)?;
                    let v_next = ((self.params.beta_2 * v.as_tensor())?
                        + ((1. - self.params.beta_2) * grad.powf(2.)?)?)?;
                    let m_hat = (&m_next / (1. - self.params.beta_1.powf(self.t)))?;

                    let delta = if rho_t > 5. {
                        let l = ((1. - self.params.beta_2.powf(self.t)).sqrt()
                            / (&v_next.sqrt()? + self.params.eps)?)?;
                        let r = ((rho_t - 4.) * (rho_t - 2.) * self.rho_inf
                            / ((self.rho_inf - 4.) * (self.rho_inf - 2.) * rho_t))
                            .sqrt();
                        (self.params.lr * r * (l * m_hat)?)?
                    } else {
                        (self.params.lr * m_hat)?
                    };
                    theta.set(&theta.sub(&(delta))?)?;
                    m.set(&m_next)?;
                    v.set(&v_next)?;
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

impl RAdam {
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
        let params = ParamsRAdam {
            lr: 0.004,
            ..Default::default()
        };
        // Now use backprop to run a linear regression between samples and get the coefficients back.
        let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
        let b = Var::new(0f32, &Device::Cpu)?;
        let mut n_sgd = RAdam::new(vec![w.clone(), b.clone()], params)?;
        assert_approx_eq!(0.004, n_sgd.learning_rate());
        n_sgd.set_learning_rate(0.002);
        assert_approx_eq!(0.002, n_sgd.learning_rate());
        Ok(())
    }

    #[test]
    fn into_inner_test() -> Result<()> {
        let params = ParamsRAdam::default();
        let w = Var::new(&[[3f32, 1.]], &Device::Cpu)?;
        let b = Var::new(-2f32, &Device::Cpu)?;
        let n_sgd = RAdam::new(vec![w.clone(), b.clone()], params)?;
        let inner = n_sgd.into_inner();
        assert_eq!(inner[0].as_tensor().to_vec2::<f32>()?, &[[3f32, 1.]]);
        assert_approx_eq!(inner[1].as_tensor().to_vec0::<f32>()?, -2_f32);
        Ok(())
    }
}
