/*!
Adadelta optimiser

Described in [ADADELTA: An Adaptive Learning Rate Method](https://arxiv.org/abs/1212.5701)

Pseudocode (including decoupling of weight decay):
$$
\\begin{aligned}
            &\\rule{110mm}{0.4pt}                                                                 \\\\
            &\\textbf{input}      : \\gamma \\text{ (lr)}, \\: \\theta_0 \\text{ (params)},
                \\: f(\\theta) \\text{ (objective)}, \\: \\rho \\text{ (decay)},
                \\: \\lambda \\text{ (weight decay)}                                                \\\\
            &\\textbf{initialize} :  v_0  \\leftarrow 0 \\: \\text{ (square avg)},
                \\: u_0 \\leftarrow 0 \\: \\text{ (accumulate variables)}                     \\\\[-1.ex]
            &\\rule{110mm}{0.4pt}                                                                 \\\\
            &\\textbf{for} \\: t=1 \\: \\textbf{to} \\: \\ldots \\: \\textbf{do}                     \\\\
            &\\hspace{5mm}g_t           \\leftarrow   \\nabla_{\\theta} f_t (\\theta_{t-1})          \\\\
            &\\hspace{5mm}\\textbf{if} \\: \\lambda \\textbf{ is } \\text{Some}                        \\\\
            &\\hspace{10mm}\\textbf{if} \\: \\textit{decoupled}                       \\\\
            &\\hspace{15mm} \\theta_t \\leftarrow \\theta_{t-1} - \\gamma \\lambda \\theta_{t-1}                    \\\\
            &\\hspace{10mm}\\textbf{else}                                                              \\\\
            &\\hspace{15mm} g_t \\leftarrow g_t + \\lambda  \\theta_{t-1}                            \\\\
            &\\hspace{5mm} v_t      \\leftarrow v_{t-1} \\rho + g^2_t (1 - \\rho)                    \\\\
            &\\hspace{5mm}\\Delta x_t    \\leftarrow   \\frac{\\sqrt{u_{t-1} +
                \\epsilon }}{ \\sqrt{v_t + \\epsilon}  }g_t \\hspace{21mm}                           \\\\
            &\\hspace{5mm} u_t  \\leftarrow   u_{t-1}  \\rho +
                 \\Delta x^2_t  (1 - \\rho)                                                        \\\\
            &\\hspace{5mm}\\theta_t      \\leftarrow   \\theta_{t-1} - \\gamma  \\Delta x_t            \\\\
            &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]
            &\\bf{return} \\:  \\theta_t                                                     \\\\[-1.ex]
            &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]
       \\end{aligned}
$$
*/

use candle_core::{Result, Var};
use candle_nn::optim::Optimizer;

use crate::{Decay, OptimParams};

/// Adadelta optimiser
///
/// Described in [ADADELTA: An Adaptive Learning Rate Method](https://arxiv.org/abs/1212.5701)
#[derive(Debug)]
pub struct Adadelta {
    vars: Vec<VarAdaDelta>,
    params: ParamsAdaDelta,
    // avg_acc: HashMap<TensorId, (Tensor, Tensor)>,
}

#[derive(Debug)]
struct VarAdaDelta {
    theta: Var,
    v: Var,
    u: Var,
}

/// Parameters for the Adadelta optimiser
#[derive(Clone, Debug)]
pub struct ParamsAdaDelta {
    /// Learning rate
    pub lr: f64,
    /// Decay
    pub rho: f64,
    /// Term added to the denominator to improve numerical stability
    pub eps: f64,
    /// Weight decay
    pub weight_decay: Option<Decay>,
}

impl Default for ParamsAdaDelta {
    fn default() -> Self {
        Self {
            lr: 1.0,
            rho: 0.9,
            weight_decay: None,
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
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let v = Var::zeros(shape, dtype, device)?;
                let u = Var::zeros(shape, dtype, device)?;
                Ok(VarAdaDelta { theta: var, v, u })
            })
            .collect::<Result<Vec<VarAdaDelta>>>()?;
        // // Err(SGDError::NoMomentum)?;
        // let mut params = params;
        // params.t = 0;
        Ok(Self {
            vars,
            params,
            // avg_acc: HashMap::new(),
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
                        let v = &var.v;
                        let u = &var.u;
                        if let Some(grad) = grads.get(theta) {
                            let grad = &(grad + (decay * theta.as_tensor())?)?;
                            let v_next = ((v.as_tensor() * self.params.rho)?
                                + (1. - self.params.rho) * grad.powf(2.)?)?;
                            let delta_x = (((u.as_tensor() + self.params.eps)?.powf(0.5)?)
                                .div(&((&v_next + self.params.eps)?.powf(0.5)?))?
                                * grad)?;
                            let u_next = ((u.as_tensor() * self.params.rho)?
                                + (1. - self.params.rho) * delta_x.powf(2.)?)?;
                            theta.set(&theta.sub(&(delta_x * self.params.lr)?)?)?;
                            v.set(&v_next)?;
                            u.set(&u_next)?;
                        }
                    }
                }
                Decay::DecoupledWeightDecay(decay) => {
                    for var in &self.vars {
                        let theta = &var.theta;
                        let v = &var.v;
                        let u = &var.u;
                        if let Some(grad) = grads.get(theta) {
                            // decoupled weight decay step
                            theta
                                .set(&(theta.as_tensor() * self.params.lr.mul_add(-decay, 1.))?)?;
                            let v_next = ((v.as_tensor() * self.params.rho)?
                                + (1. - self.params.rho) * grad.powf(2.)?)?;
                            let delta_x = (((u.as_tensor() + self.params.eps)?.powf(0.5)?)
                                .div(&((&v_next + self.params.eps)?.powf(0.5)?))?
                                * grad)?;
                            let u_next = ((u.as_tensor() * self.params.rho)?
                                + (1. - self.params.rho) * delta_x.powf(2.)?)?;
                            theta.set(&theta.sub(&(delta_x * self.params.lr)?)?)?;
                            v.set(&v_next)?;
                            u.set(&u_next)?;
                        }
                    }
                }
            }
        } else {
            for var in &self.vars {
                let theta = &var.theta;
                let v = &var.v;
                let u = &var.u;
                if let Some(grad) = grads.get(theta) {
                    let v_next = ((v.as_tensor() * self.params.rho)?
                        + (1. - self.params.rho) * grad.powf(2.)?)?;
                    let delta_x = (((u.as_tensor() + self.params.eps)?.powf(0.5)?)
                        .div(&((&v_next + self.params.eps)?.powf(0.5)?))?
                        * grad)?;
                    let u_next = ((u.as_tensor() * self.params.rho)?
                        + (1. - self.params.rho) * delta_x.powf(2.)?)?;
                    theta.set(&theta.sub(&(delta_x * self.params.lr)?)?)?;
                    v.set(&v_next)?;
                    u.set(&u_next)?;
                }
            }
        }

        Ok(())
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }
}

impl OptimParams for Adadelta {
    fn params(&self) -> &Self::Config {
        &self.params
    }

    fn set_params(&mut self, config: Self::Config) {
        self.params = config;
    }
}

impl Adadelta {
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
        let params = ParamsAdaDelta {
            lr: 0.004,
            ..Default::default()
        };
        // Now use backprop to run a linear regression between samples and get the coefficients back.
        let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
        let b = Var::new(0f32, &Device::Cpu)?;
        let mut optim = Adadelta::new(vec![w.clone(), b.clone()], params)?;
        assert_approx_eq!(0.004, optim.learning_rate());
        optim.set_learning_rate(0.002);
        assert_approx_eq!(0.002, optim.learning_rate());
        Ok(())
    }

    #[test]
    fn into_inner_test() -> Result<()> {
        let params = ParamsAdaDelta::default();
        let w = Var::new(&[[3f32, 1.]], &Device::Cpu)?;
        let b = Var::new(-2f32, &Device::Cpu)?;
        let optim = Adadelta::new(vec![w.clone(), b.clone()], params)?;
        let inner = optim.into_inner();
        assert_eq!(inner[0].as_tensor().to_vec2::<f32>()?, &[[3f32, 1.]]);
        assert_approx_eq!(inner[1].as_tensor().to_vec0::<f32>()?, -2_f32);
        Ok(())
    }
}
