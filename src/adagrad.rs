/*!
Adagrad optimiser

Described in [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://jmlr.org/papers/v12/duchi11a.html)

Pseudocode (including decoupling of weight decay):

$$
\\begin{aligned}
            &\\rule{110mm}{0.4pt}                                                                 \\\\
            &\\textbf{input}      : \\gamma \\text{ (lr)}, \\: \\theta_0 \\text{ (params)}, \\: f(\\theta)
                \\text{ (objective)}, \\: \\lambda \\text{ (weight decay)},                          \\\\
            &\\hspace{12mm}    \\tau \\text{ (initial accumulator value)}, \\: \\eta\\text{ (lr decay)}\\\\
            &\\textbf{initialize} :  statesum_0 \\leftarrow 0                             \\\\[-1.ex]
            &\\rule{110mm}{0.4pt}                                                                 \\\\
            &\\textbf{for} \\: t=1 \\: \\textbf{to} \\: \\ldots \\: \\textbf{do}                         \\\\
            &\\hspace{5mm}g_t           \\leftarrow   \\nabla_{\\theta} f_t (\\theta_{t-1})           \\\\
            &\\hspace{5mm} \\tilde{\\gamma}    \\leftarrow \\gamma / (1 +(t-1) \\eta)                  \\\\
            &\\hspace{5mm}\\textbf{if} \\: \\lambda \\textbf{ is } \\text{Some}                        \\\\
            &\\hspace{10mm}\\textbf{if} \\: \\textit{decoupled}                       \\\\
            &\\hspace{15mm} \\theta_t \\leftarrow \\theta_{t-1} - \\gamma \\lambda \\theta_{t-1}                    \\\\
            &\\hspace{10mm}\\textbf{else}                                                              \\\\
            &\\hspace{15mm} g_t \\leftarrow g_t + \\lambda  \\theta_{t-1}                            \\\\
            &\\hspace{5mm}statesum_t  \\leftarrow  statesum_{t-1} + g^2_t                      \\\\
            &\\hspace{5mm}\\theta_t \\leftarrow
                \\theta_{t-1}- \\tilde{\\gamma} \\frac{g_t}{\\sqrt{statesum_t}+\\epsilon}            \\\\
            &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]
            &\\bf{return} \\:  \\theta_t                                                     \\\\[-1.ex]
            &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]
       \\end{aligned}
$$



*/

use candle_core::{Result, Var};
use candle_nn::optim::Optimizer;

use crate::{Decay, OptimParams};

/// Adagrad optimiser
///
/// Described in [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://jmlr.org/papers/v12/duchi11a.html)
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

/// Parameters for the Adagrad optimiser
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ParamsAdaGrad {
    /// Learning rate
    pub lr: f64,
    /// Learning rate decay
    pub lr_decay: f64,
    /// Initial value of accumulator
    pub initial_acc: f64,
    /// weight decay
    pub weight_decay: Option<Decay>,
    /// term added to the denominator to improve numerical stability
    pub eps: f64,
}

impl Default for ParamsAdaGrad {
    fn default() -> Self {
        Self {
            lr: 0.01,
            lr_decay: 0.0,
            initial_acc: 0.0,
            weight_decay: None,
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
        if let Some(decay) = self.params.weight_decay {
            match decay {
                Decay::WeightDecay(decay) => {
                    for var in &self.vars {
                        let theta = &var.theta;
                        let sum = &var.sum;
                        if let Some(grad) = grads.get(theta) {
                            let gamma_tilde =
                                self.params.lr / self.t.mul_add(self.params.lr_decay, 1.);
                            let grad = &(grad + (decay * theta.as_tensor())?)?;
                            let current_sum = (sum.as_tensor() + grad.powf(2.)?)?;
                            let change = (gamma_tilde
                                * (grad.div(&(current_sum.powf(0.5)? + self.params.eps)?))?)?;
                            sum.set(&current_sum)?;
                            theta.set(&theta.sub(&change)?)?;
                        }
                    }
                }
                Decay::DecoupledWeightDecay(decay) => {
                    for var in &self.vars {
                        let theta = &var.theta;
                        let sum = &var.sum;
                        if let Some(grad) = grads.get(theta) {
                            // decoupled weight decay step
                            theta
                                .set(&(theta.as_tensor() * self.params.lr.mul_add(-decay, 1.))?)?;
                            let gamma_tilde =
                                self.params.lr / self.t.mul_add(self.params.lr_decay, 1.);
                            let current_sum = (sum.as_tensor() + grad.powf(2.)?)?;
                            let change = (gamma_tilde
                                * (grad.div(&(current_sum.powf(0.5)? + self.params.eps)?))?)?;
                            sum.set(&current_sum)?;
                            theta.set(&theta.sub(&change)?)?;
                        }
                    }
                }
            }
        } else {
            for var in &self.vars {
                let theta = &var.theta;
                let sum = &var.sum;
                if let Some(grad) = grads.get(theta) {
                    let gamma_tilde = self.params.lr / self.t.mul_add(self.params.lr_decay, 1.);
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

impl OptimParams for Adagrad {
    fn params(&self) -> &Self::Config {
        &self.params
    }

    fn set_params(&mut self, config: Self::Config) {
        self.params = config;
    }
}

impl Adagrad {
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
        let params = ParamsAdaGrad {
            lr: 0.004,
            ..Default::default()
        };
        // Now use backprop to run a linear regression between samples and get the coefficients back.
        let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
        let b = Var::new(0f32, &Device::Cpu)?;
        let mut optim = Adagrad::new(vec![w.clone(), b.clone()], params)?;
        assert_approx_eq!(0.004, optim.learning_rate());
        optim.set_learning_rate(0.002);
        assert_approx_eq!(0.002, optim.learning_rate());
        Ok(())
    }

    #[test]
    fn into_inner_test() -> Result<()> {
        let params = ParamsAdaGrad::default();
        let w = Var::new(&[[3f32, 1.]], &Device::Cpu)?;
        let b = Var::new(-2f32, &Device::Cpu)?;
        let optim = Adagrad::new(vec![w.clone(), b.clone()], params)?;
        let inner = optim.into_inner();
        assert_eq!(inner[0].as_tensor().to_vec2::<f32>()?, &[[3f32, 1.]]);
        assert_approx_eq!(inner[1].as_tensor().to_vec0::<f32>()?, -2_f32);
        Ok(())
    }
}
