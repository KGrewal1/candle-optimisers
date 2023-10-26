use candle_core::{Result, Var};
use candle_nn::optim::Optimizer;

/// Adam optimizer with Nesterov momentum
///
/// Described in <https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ>
///
/// For pseudocde see <https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam>

#[derive(Debug)]
pub struct NAdam {
    vars: Vec<VarNAdam>,
    params: ParamsNAdam,
    mu_t: f64,
    mu_t2: f64,
    prod: f64,
    prod2: f64,
    t: f64,
}

#[derive(Debug)]
struct VarNAdam {
    theta: Var,
    m: Var,
    v: Var,
}

#[derive(Debug)]
pub struct ParamsNAdam {
    pub lr: f64,
    pub beta_1: f64,
    pub beta_2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub momentum_decay: f64,
    pub decoupled_weight_decay: bool,
}

impl Default for ParamsNAdam {
    fn default() -> Self {
        Self {
            lr: 0.002,
            beta_1: 0.9,
            beta_2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum_decay: 0.004,
            decoupled_weight_decay: false,
        }
    }
}

impl Optimizer for NAdam {
    type Config = ParamsNAdam;

    fn new(vars: Vec<Var>, params: ParamsNAdam) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let m = Var::zeros(shape, dtype, device)?;
                let v = Var::zeros(shape, dtype, device)?;
                Ok(VarNAdam { theta: var, m, v })
            })
            .collect::<Result<Vec<VarNAdam>>>()?;
        // // Err(SGDError::NoMomentum)?;
        // let mut params = params;
        // params.t = 0;
        let t = 1.;
        let mu_t2 = params.beta_1 * (1. - 0.5 * (0.96_f64.powf(t * params.momentum_decay)));
        Ok(Self {
            vars,
            params,
            t: 1.,
            mu_t: 1.,
            mu_t2,
            prod: 1.,
            prod2: mu_t2,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        let mu_t = self.mu_t2;
        let mu_t2 = self.params.beta_1
            * (1. - 0.5 * (0.96_f64.powf((self.t + 1.) * self.params.momentum_decay)));
        let prod = self.prod2;
        let prod2 = prod * mu_t2;
        self.mu_t = mu_t;
        self.mu_t2 = mu_t2;
        self.prod = prod;
        self.prod2 = prod2;
        // println!("prod {}", prod);
        for var in &self.vars {
            let theta = &var.theta;
            let m = &var.m;
            let v = &var.v;
            if let Some(grad) = grads.get(theta) {
                if self.params.weight_decay == 0. {
                    let m_next = ((self.params.beta_1 * m.as_tensor())?
                        + ((1. - self.params.beta_1) * grad)?)?;
                    let v_next = ((self.params.beta_2 * v.as_tensor())?
                        + ((1. - self.params.beta_2) * grad.powf(2.)?)?)?;
                    let m_hat = (((mu_t2 / (1. - prod2)) * &m_next)?
                        + (((1. - mu_t) / (1. - prod)) * grad)?)?;
                    let v_hat = (&v_next / (1. - self.params.beta_2.powf(self.t)))?;
                    let delta =
                        (m_hat * self.params.lr)?.div(&(v_hat.powf(0.5)? + self.params.eps)?)?;
                    theta.set(&theta.sub(&(delta))?)?;
                    m.set(&m_next)?;
                    v.set(&v_next)?;
                } else if self.params.decoupled_weight_decay {
                    theta.set(
                        &(theta.as_tensor() * (1. - self.params.lr * self.params.weight_decay))?,
                    )?;
                    let m_next = ((self.params.beta_1 * m.as_tensor())?
                        + ((1. - self.params.beta_1) * grad)?)?;
                    let v_next = ((self.params.beta_2 * v.as_tensor())?
                        + ((1. - self.params.beta_2) * grad.powf(2.)?)?)?;
                    let m_hat = (((mu_t2 / (1. - prod2)) * &m_next)?
                        + (((1. - mu_t) / (1. - prod)) * grad)?)?;
                    let v_hat = (&v_next / (1. - self.params.beta_2.powf(self.t)))?;
                    let delta =
                        (m_hat * self.params.lr)?.div(&(v_hat.powf(0.5)? + self.params.eps)?)?;
                    theta.set(&theta.sub(&(delta))?)?;
                    m.set(&m_next)?;
                    v.set(&v_next)?;
                } else {
                    let grad = &(grad + (self.params.weight_decay * theta.as_tensor())?)?;
                    let m_next = ((self.params.beta_1 * m.as_tensor())?
                        + ((1. - self.params.beta_1) * grad)?)?;
                    let v_next = ((self.params.beta_2 * v.as_tensor())?
                        + ((1. - self.params.beta_2) * grad.powf(2.)?)?)?;
                    let m_hat = (((mu_t2 / (1. - prod2)) * &m_next)?
                        + (((1. - mu_t) / (1. - prod)) * grad)?)?;
                    let v_hat = (&v_next / (1. - self.params.beta_2.powf(self.t)))?;
                    let delta =
                        (m_hat * self.params.lr)?.div(&(v_hat.powf(0.5)? + self.params.eps)?)?;
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

impl NAdam {
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

        let params = ParamsNAdam {
            lr: 0.004,
            ..Default::default()
        };
        // Now use backprop to run a linear regression between samples and get the coefficients back.
        let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
        let b = Var::new(0f32, &Device::Cpu)?;
        let mut n_sgd = NAdam::new(vec![w.clone(), b.clone()], params)?;
        assert_approx_eq!(0.004, n_sgd.learning_rate());
        n_sgd.set_learning_rate(0.002);
        assert_approx_eq!(0.002, n_sgd.learning_rate());
        Ok(())
    }
}
