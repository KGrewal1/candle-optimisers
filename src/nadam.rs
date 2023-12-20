/*!
NAdam optimiser: Adam with Nesterov momentum

Described in [Incorporating Nesterov Momentum into Adam](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ)

Pseudocode (including decoupling of weight decay):

$$
\\begin{aligned}
    &\\rule{110mm}{0.4pt}                                                                 \\\\
    &\\textbf{input}      : \\gamma_t \\text{ (lr)}, \\: \\beta_1,\\beta_2 \\text{ (betas)},
        \\: \\theta_0 \\text{ (params)}, \\: f(\\theta) \\text{ (objective)}                   \\\\
    &\\hspace{12mm} \\: \\lambda \\text{ (weight decay)}, \\:\\psi \\text{ (momentum decay)}    \\\\
    &\\textbf{initialize} :  m_0 \\leftarrow 0 \\text{ ( first moment)},
        v_0 \\leftarrow 0 \\text{ ( second moment)}                                 \\\\[-1.ex]
    &\\rule{110mm}{0.4pt}                                                                 \\\\
    &\\textbf{for} \\: t=1 \\: \\textbf{to} \\: \\ldots \\: \\textbf{do}                         \\\\
    &\\hspace{5mm}g_t           \\leftarrow   \\nabla_{\\theta} f_t (\\theta_{t-1})           \\\\
    &\\hspace{5mm} \\theta_t \\leftarrow \\theta_{t-1}                                       \\\\
    &\\hspace{5mm}\\textbf{if} \\: \\lambda \\textbf{ is } \\text{Some}                        \\\\
    &\\hspace{10mm}\\textbf{if} \\: \\textit{decoupled}                       \\\\
    &\\hspace{15mm} \\theta_t \\leftarrow \\theta_{t-1} - \\gamma \\lambda \\theta_{t-1}                    \\\\
    &\\hspace{10mm}\\textbf{else}                                                              \\\\
    &\\hspace{15mm} g_t \\leftarrow g_t + \\lambda  \\theta_{t-1}                            \\\\
    &\\hspace{5mm} \\mu_t \\leftarrow \\beta_1 \\big(1 - \\frac{1}{2}  0.96^{t \\psi} \\big)     \\\\
    &\\hspace{5mm} \\mu_{t+1} \\leftarrow \\beta_1 \\big(1 - \\frac{1}{2} 0.96^{(t+1)\\psi}\\big)\\\\
    &\\hspace{5mm}m_t           \\leftarrow   \\beta_1 m_{t-1} + (1 - \\beta_1) g_t          \\\\
    &\\hspace{5mm}v_t           \\leftarrow   \\beta_2 v_{t-1} + (1-\\beta_2) g^2_t          \\\\
    &\\hspace{5mm}\\widehat{m_t} \\leftarrow \\mu_{t+1} m_t/(1-\\prod_{i=1}^{t+1}\\mu_i)\\\\[-1.ex]
    & \\hspace{11mm} + (1-\\mu_t) g_t /(1-\\prod_{i=1}^{t} \\mu_{i})                         \\\\
    &\\hspace{5mm}\\widehat{v_t} \\leftarrow   v_t/\\big(1-\\beta_2^t \\big)                   \\\\
    &\\hspace{5mm}\\theta_t \\leftarrow \\theta_t - \\gamma \\widehat{m_t}/
        \\big(\\sqrt{\\widehat{v_t}} + \\epsilon \\big)                                       \\\\
    &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]
    &\\bf{return} \\:  \\theta_t                                                     \\\\[-1.ex]
    &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]
\\end{aligned}
$$
*/

use candle_core::{Result, Var};
use candle_nn::optim::Optimizer;

use crate::{Decay, OptimParams};

/// Adam optimiser with Nesterov momentum
///
/// Described in <https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ>
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

/// Parameters for The NAdam optimiser
#[derive(Debug)]
pub struct ParamsNAdam {
    /// Learning rate
    pub lr: f64,
    /// Coefficient for moving average of first moment
    pub beta_1: f64,
    /// Coefficient for moving average of second moment
    pub beta_2: f64,
    /// Term added to denominator to improve numerical stability
    pub eps: f64,
    /// Weight decay
    pub weight_decay: Option<Decay>,
    /// Momentum decay
    pub momentum_decay: f64,
}

impl Default for ParamsNAdam {
    fn default() -> Self {
        Self {
            lr: 0.002,
            beta_1: 0.9,
            beta_2: 0.999,
            eps: 1e-8,
            weight_decay: None,
            momentum_decay: 0.004,
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
        let mu_t2 = params.beta_1 * 0.5f64.mul_add(-(0.96_f64.powf(t * params.momentum_decay)), 1.);
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
            * 0.5f64.mul_add(
                -(0.96_f64.powf((self.t + 1.) * self.params.momentum_decay)),
                1.,
            );
        let prod = self.prod2;
        let prod2 = prod * mu_t2;
        self.mu_t = mu_t;
        self.mu_t2 = mu_t2;
        self.prod = prod;
        self.prod2 = prod2;
        // println!("prod {}", prod);

        if let Some(decay) = self.params.weight_decay {
            match decay {
                Decay::WeightDecay(decay) => {
                    for var in &self.vars {
                        let theta = &var.theta;
                        let m = &var.m;
                        let v = &var.v;
                        if let Some(grad) = grads.get(theta) {
                            let grad = &(grad + (decay * theta.as_tensor())?)?;
                            let m_next = ((self.params.beta_1 * m.as_tensor())?
                                + ((1. - self.params.beta_1) * grad)?)?;
                            let v_next = ((self.params.beta_2 * v.as_tensor())?
                                + ((1. - self.params.beta_2) * grad.powf(2.)?)?)?;
                            let m_hat = (((mu_t2 / (1. - prod2)) * &m_next)?
                                + (((1. - mu_t) / (1. - prod)) * grad)?)?;
                            let v_hat = (&v_next / (1. - self.params.beta_2.powf(self.t)))?;
                            let delta = (m_hat * self.params.lr)?
                                .div(&(v_hat.powf(0.5)? + self.params.eps)?)?;
                            theta.set(&theta.sub(&(delta))?)?;
                            m.set(&m_next)?;
                            v.set(&v_next)?;
                        }
                    }
                }
                Decay::DecoupledWeightDecay(decay) => {
                    for var in &self.vars {
                        let theta = &var.theta;
                        let m = &var.m;
                        let v = &var.v;
                        if let Some(grad) = grads.get(theta) {
                            theta
                                .set(&(theta.as_tensor() * self.params.lr.mul_add(-decay, 1.))?)?;
                            let m_next = ((self.params.beta_1 * m.as_tensor())?
                                + ((1. - self.params.beta_1) * grad)?)?;
                            let v_next = ((self.params.beta_2 * v.as_tensor())?
                                + ((1. - self.params.beta_2) * grad.powf(2.)?)?)?;
                            let m_hat = (((mu_t2 / (1. - prod2)) * &m_next)?
                                + (((1. - mu_t) / (1. - prod)) * grad)?)?;
                            let v_hat = (&v_next / (1. - self.params.beta_2.powf(self.t)))?;
                            let delta = (m_hat * self.params.lr)?
                                .div(&(v_hat.powf(0.5)? + self.params.eps)?)?;
                            theta.set(&theta.sub(&(delta))?)?;
                            m.set(&m_next)?;
                            v.set(&v_next)?;
                        }
                    }
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

impl OptimParams for NAdam {
    fn params(&self) -> &Self::Config {
        &self.params
    }

    fn set_params(&mut self, config: Self::Config) {
        self.params = config;
    }
}

impl NAdam {
    /// Return the vars being optimised
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

    #[test]
    fn into_inner_test() -> Result<()> {
        let params = ParamsNAdam::default();
        let w = Var::new(&[[3f32, 1.]], &Device::Cpu)?;
        let b = Var::new(-2f32, &Device::Cpu)?;
        let n_sgd = NAdam::new(vec![w.clone(), b.clone()], params)?;
        let inner = n_sgd.into_inner();
        assert_eq!(inner[0].as_tensor().to_vec2::<f32>()?, &[[3f32, 1.]]);
        assert_approx_eq!(inner[1].as_tensor().to_vec0::<f32>()?, -2_f32);
        Ok(())
    }
}
