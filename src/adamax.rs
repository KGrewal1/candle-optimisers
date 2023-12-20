/*!
Adamax optimiser

An Adam optimiser based on infinity norm, described in [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

Pseudocode (including decoupling of weight decay):

$$
\\begin{aligned}
    &\\rule{110mm}{0.4pt}                                                                 \\\\
    &\\textbf{input}      : \\gamma \\text{ (lr)}, \\beta_1, \\beta_2
        \\text{ (betas)},\\theta_0 \\text{ (params)},f(\\theta) \\text{ (objective)},
        \\: \\lambda \\text{ (weight decay)},                                                \\\\
    &\\hspace{13mm}    \\epsilon \\text{ (epsilon)}                                          \\\\
    &\\textbf{initialize} :  m_0 \\leftarrow 0 \\text{ ( first moment)},
        u_0 \\leftarrow 0 \\text{ ( infinity norm)}                                 \\\\[-1.ex]
    &\\rule{110mm}{0.4pt}                                                                 \\\\
    &\\textbf{for} \\: t=1 \\: \\textbf{to} \\: \\ldots \\: \\textbf{do}                         \\\\
    &\\hspace{5mm}g_t           \\leftarrow   \\nabla_{\\theta} f_t (\\theta_{t-1})           \\\\
    &\\hspace{5mm}\\textbf{if} \\: \\lambda \\textbf{ is } \\text{Some}                        \\\\
    &\\hspace{10mm}\\textbf{if} \\: \\textit{decoupled}                       \\\\
    &\\hspace{15mm} \\theta_t \\leftarrow \\theta_{t-1} - \\gamma \\lambda \\theta_{t-1}                    \\\\
    &\\hspace{10mm}\\textbf{else}                                                              \\\\
    &\\hspace{15mm} g_t \\leftarrow g_t + \\lambda  \\theta_{t-1}                            \\\\
    &\\hspace{5mm}m_t      \\leftarrow   \\beta_1 m_{t-1} + (1 - \\beta_1) g_t               \\\\
    &\\hspace{5mm}u_t      \\leftarrow   \\mathrm{max}(\\beta_2 u_{t-1}, |g_{t}|+\\epsilon)   \\\\
    &\\hspace{5mm}\\theta_t \\leftarrow \\theta_{t-1} - \\frac{\\gamma m_t}{(1-\\beta^t_1) u_t} \\\\
    &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]
    &\\bf{return} \\:  \\theta_t                                                     \\\\[-1.ex]
    &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]
\\end{aligned}
$$
*/

use candle_core::{Result, Var};
use candle_nn::optim::Optimizer;

use crate::{Decay, OptimParams};

/// Adamax optimiser
///
/// An Adam optimiser based on infinity norm, described in [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

#[derive(Debug)]
pub struct Adamax {
    vars: Vec<VarAdaMax>,
    params: ParamsAdaMax,
    t: f64,
}

#[derive(Debug)]
struct VarAdaMax {
    theta: Var,
    m: Var,
    u: Var,
}

/// Parameters for the Adamax optimiser
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ParamsAdaMax {
    /// Learning rate
    pub lr: f64,
    /// Coefficient for moving average of first moment
    pub beta_1: f64,
    /// Coefficient for moving average of second moment
    pub beta_2: f64,
    /// Weight decay
    pub weight_decay: Option<Decay>,
    /// Term added to denominator to improve numerical stability
    pub eps: f64,
}

impl Default for ParamsAdaMax {
    fn default() -> Self {
        Self {
            lr: 1.0,
            beta_1: 0.9,
            beta_2: 0.999,
            weight_decay: None,
            eps: 1e-8,
        }
    }
}

impl Optimizer for Adamax {
    type Config = ParamsAdaMax;

    fn new(vars: Vec<Var>, params: ParamsAdaMax) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let m = Var::zeros(shape, dtype, device)?;
                let u = Var::zeros(shape, dtype, device)?;
                Ok(VarAdaMax { theta: var, m, u })
            })
            .collect::<Result<Vec<VarAdaMax>>>()?;
        // // Err(SGDError::NoMomentum)?;
        // let mut params = params;
        // params.t = 0;
        Ok(Self {
            vars,
            params,
            t: 1.,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        if let Some(decay) = self.params.weight_decay {
            match decay {
                Decay::WeightDecay(decay) => {
                    for var in &self.vars {
                        let theta = &var.theta;
                        let m = &var.m;
                        let u = &var.u;
                        if let Some(grad) = grads.get(theta) {
                            let grad = &(grad + (decay * theta.as_tensor())?)?;
                            let m_next = ((self.params.beta_1 * m.as_tensor())?
                                + (1. - self.params.beta_1) * grad)?;
                            let u_next = (self.params.beta_2 * u.as_tensor())?
                                .maximum(&(grad.abs()? + self.params.eps)?)?;
                            let delta = (&m_next * self.params.lr)?
                                .div(&(&u_next * (1. - self.params.beta_1.powf(self.t)))?)?;
                            theta.set(&theta.sub(&(delta))?)?;
                            m.set(&m_next)?;
                            u.set(&u_next)?;
                        }
                    }
                }
                Decay::DecoupledWeightDecay(decay) => {
                    for var in &self.vars {
                        let theta = &var.theta;
                        let m = &var.m;
                        let u = &var.u;
                        if let Some(grad) = grads.get(theta) {
                            // decoupled weight decay step
                            theta
                                .set(&(theta.as_tensor() * self.params.lr.mul_add(-decay, 1.))?)?;
                            let m_next = ((self.params.beta_1 * m.as_tensor())?
                                + (1. - self.params.beta_1) * grad)?;
                            let u_next = (self.params.beta_2 * u.as_tensor())?
                                .maximum(&(grad.abs()? + self.params.eps)?)?;
                            let delta = (&m_next * self.params.lr)?
                                .div(&(&u_next * (1. - self.params.beta_1.powf(self.t)))?)?;
                            theta.set(&theta.sub(&(delta))?)?;
                            m.set(&m_next)?;
                            u.set(&u_next)?;
                        }
                    }
                }
            }
        } else {
            for var in &self.vars {
                let theta = &var.theta;
                let m = &var.m;
                let u = &var.u;
                if let Some(grad) = grads.get(theta) {
                    let m_next =
                        ((self.params.beta_1 * m.as_tensor())? + (1. - self.params.beta_1) * grad)?;
                    let u_next = (self.params.beta_2 * u.as_tensor())?
                        .maximum(&(grad.abs()? + self.params.eps)?)?;
                    let delta = (&m_next * self.params.lr)?
                        .div(&(&u_next * (1. - self.params.beta_1.powf(self.t)))?)?;
                    theta.set(&theta.sub(&(delta))?)?;
                    m.set(&m_next)?;
                    u.set(&u_next)?;
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

impl OptimParams for Adamax {
    fn params(&self) -> &Self::Config {
        &self.params
    }

    fn set_params(&mut self, config: Self::Config) {
        self.params = config;
    }
}

impl Adamax {
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
        let params = ParamsAdaMax {
            lr: 0.004,
            ..Default::default()
        };
        // Now use backprop to run a linear regression between samples and get the coefficients back.
        let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
        let b = Var::new(0f32, &Device::Cpu)?;
        let mut optim = Adamax::new(vec![w.clone(), b.clone()], params)?;
        assert_approx_eq!(0.004, optim.learning_rate());
        optim.set_learning_rate(0.002);
        assert_approx_eq!(0.002, optim.learning_rate());
        Ok(())
    }

    #[test]
    fn into_inner_test() -> Result<()> {
        let params = ParamsAdaMax::default();
        let w = Var::new(&[[3f32, 1.]], &Device::Cpu)?;
        let b = Var::new(-2f32, &Device::Cpu)?;
        let optim = Adamax::new(vec![w.clone(), b.clone()], params)?;
        let inner = optim.into_inner();
        assert_eq!(inner[0].as_tensor().to_vec2::<f32>()?, &[[3f32, 1.]]);
        assert_approx_eq!(inner[1].as_tensor().to_vec0::<f32>()?, -2_f32);
        Ok(())
    }

    #[test]
    fn params_test() -> Result<()> {
        let params = ParamsAdaMax {
            lr: 0.004,
            ..Default::default()
        };
        // Now use backprop to run a linear regression between samples and get the coefficients back.
        let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
        let b = Var::new(0f32, &Device::Cpu)?;
        let mut optim = Adamax::new(vec![w.clone(), b.clone()], params.clone())?;
        assert_eq!(params, optim.params().clone());
        let new_params = ParamsAdaMax {
            lr: 0.002,
            ..Default::default()
        };
        optim.set_params(new_params.clone());
        assert_eq!(new_params, optim.params().clone());
        Ok(())
    }
}
