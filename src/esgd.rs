/*!
   Stochastic Gradient Descent

   This incoporates Nesterov and classical momentum as well as weight decay and decoupled weight decay
   (as described as SGDW in [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101))

$$
\\begin{aligned}
    &\\rule{110mm}{0.4pt}                                                                 \\\\
     &\\textbf{input}      : \\gamma \\text{ (lr)}, \\: \\theta_0 \\text{ (params)}, \\: f(\\theta)
        \\text{ (objective)}, \\: \\lambda \\text{ (weight decay)},                          \\\\
   &\\hspace{13mm} \\:\\mu \\text{ (momentum)}, \\:\\tau \\text{ (dampening)}          \\\\[-1.ex]
    &\\rule{110mm}{0.4pt}                                                                 \\\\
    &\\textbf{for} \\: t=1 \\: \\textbf{to} \\: \\ldots \\: \\textbf{do}                         \\\\
    &\\hspace{5mm}g_t           \\leftarrow   \\nabla_{\\theta} f_t (\\theta_{t-1})           \\\\
    &\\hspace{5mm}\\textbf{if} \\: \\lambda \\textbf{ is } \\text{Some}                        \\\\
    &\\hspace{10mm}\\textbf{if} \\: \\textit{decoupled}                       \\\\
    &\\hspace{15mm} \\theta_t \\leftarrow \\theta_{t-1} - \\gamma \\lambda \\theta_{t-1}                    \\\\
    &\\hspace{10mm}\\textbf{else}                                                              \\\\
    &\\hspace{15mm} g_t \\leftarrow g_t + \\lambda  \\theta_{t-1}                            \\\\
    &\\hspace{5mm}\\textbf{if} \\: \\mu \\textbf{ is } \\text{Some}                        \\\\
    &\\hspace{10mm}\\textbf{if} \\: t>1                      \\\\
    &\\hspace{15mm} b_t \\leftarrow \\mu b_{t-1} + (1-\\tau)g_{t}                   \\\\
    &\\hspace{10mm}\\textbf{else}                                                              \\\\
    &\\hspace{15mm} b_t \\leftarrow g_{t}                                    \\\\
     &\\hspace{10mm}\\textbf{if} \\: \\textit{nesterov}                       \\\\
    &\\hspace{15mm} g_t \\leftarrow g_t + \\mu b_t                   \\\\
    &\\hspace{10mm}\\textbf{else}                                                              \\\\
    &\\hspace{15mm} g_t \\leftarrow b_t                           \\\\
    &\\hspace{5mm}\\theta_t \\leftarrow \\theta_{t-1} - \\gamma g_t \\\\
    &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]
    &\\bf{return} \\:  \\theta_t                                                     \\\\[-1.ex]
    &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]
\\end{aligned}
$$

*/

use candle_core::{Result, Var};
use candle_nn::optim::Optimizer;

use crate::{Decay, Momentum, OptimParams};

/// Optimizer for Stochastic Gradient Descent with momentum.
#[derive(Debug)]
pub struct SGD {
    vars: Vec<VarSGD>,
    params: ParamsSGD,
}

#[derive(Debug)]
struct VarSGD {
    theta: Var,
    b: Option<Var>,
}

/// Parameters for SGD
#[derive(Clone, Debug)]
pub struct ParamsSGD {
    /// Learning rate
    pub lr: f64,
    /// Weight decay
    pub weight_decay: Option<Decay>,
    /// Momentum
    pub momentum: Option<Momentum>,
    /// Dampening
    pub dampening: f64,
}

impl Default for ParamsSGD {
    fn default() -> Self {
        Self {
            lr: 0.1,
            weight_decay: None,
            momentum: None, //Momentum::Classical(0.1)
            dampening: 0.0,
            // nesterov: false,
        }
    }
}

impl Optimizer for SGD {
    type Config = ParamsSGD;

    fn new(vars: Vec<Var>, params: ParamsSGD) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| VarSGD {
                theta: var,
                b: None,
            })
            .collect::<Vec<VarSGD>>();
        // Err(SGDError::NoMomentum)?;
        Ok(Self { vars, params })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    #[allow(clippy::too_many_lines)]
    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        if let Some(momentum) = self.params.momentum {
            match momentum {
                Momentum::Classical(momentum) => {
                    if let Some(decay) = self.params.weight_decay {
                        match decay {
                            Decay::WeightDecay(decay) => {
                                for var in &mut self.vars {
                                    let theta = &var.theta;
                                    // let prev_step = var.b;
                                    if let Some(grad) = grads.get(theta) {
                                        let grad = &(grad + (decay * theta.as_tensor())?)?;
                                        if let Some(prev_step) = &(var.b) {
                                            // println!("Exists");
                                            // bt​←μbt−1​+(1−τ)gt
                                            let bt = ((prev_step.as_tensor() * momentum)?
                                                + (1. - self.params.dampening) * (grad))?;

                                            // if not nesterov gt = bt
                                            theta.set(&theta.sub(&(&bt * self.params.lr)?)?)?;
                                            // println!("Momentum {}", bt);
                                            prev_step.set(&bt)?;
                                        } else {
                                            // println!("Doesn't Exist");
                                            // bt​←μbt−1​+(1−τ)gt
                                            // if there is no history bt = gt = grad with no weight_decay
                                            let bt = grad.clone(); // clone must occur invariably due to need to store in hashmap

                                            // if not nesterov gt = bt
                                            theta.set(&theta.sub(&(&bt * self.params.lr)?)?)?;
                                            // println!("Momentum {}", bt);
                                            var.b = Some(Var::from_tensor(&bt)?);
                                        };
                                    }
                                }
                            }
                            Decay::DecoupledWeightDecay(decay) => {
                                for var in &mut self.vars {
                                    let theta = &var.theta;
                                    // let prev_step = var.b;
                                    if let Some(grad) = grads.get(theta) {
                                        // decoupled weight decay step
                                        theta.set(
                                            &(theta.as_tensor()
                                                * self.params.lr.mul_add(-decay, 1.))?,
                                        )?;
                                        if let Some(prev_step) = &(var.b) {
                                            // println!("Exists");
                                            // bt​←μbt−1​+(1−τ)gt
                                            let bt = ((prev_step.as_tensor() * momentum)?
                                                + (1. - self.params.dampening) * (grad))?;

                                            // if not nesterov gt = bt
                                            theta.set(&theta.sub(&(&bt * self.params.lr)?)?)?;
                                            // println!("Momentum {}", bt);
                                            prev_step.set(&bt)?;
                                        } else {
                                            // println!("Doesn't Exist");
                                            // bt​←μbt−1​+(1−τ)gt
                                            // if there is no history bt = gt = grad with no weight_decay
                                            let bt = grad.clone(); // clone must occur invariably due to need to store in hashmap

                                            // if not nesterov gt = bt
                                            theta.set(&theta.sub(&(&bt * self.params.lr)?)?)?;
                                            // println!("Momentum {}", bt);
                                            var.b = Some(Var::from_tensor(&bt)?);
                                        };
                                    }
                                }
                            }
                        }
                    } else {
                        for var in &mut self.vars {
                            let theta = &var.theta;
                            // let prev_step = var.b;
                            if let Some(grad) = grads.get(theta) {
                                if let Some(prev_step) = &(var.b) {
                                    // println!("Exists");
                                    // bt​←μbt−1​+(1−τ)gt
                                    let bt = ((prev_step.as_tensor() * momentum)?
                                        + (1. - self.params.dampening) * (grad))?;

                                    // if not nesterov gt = bt
                                    theta.set(&theta.sub(&(&bt * self.params.lr)?)?)?;
                                    // println!("Momentum {}", bt);
                                    prev_step.set(&bt)?;
                                } else {
                                    // println!("Doesn't Exist");
                                    // bt​←μbt−1​+(1−τ)gt
                                    // if there is no history bt = gt = grad with no weight_decay
                                    let bt = grad.clone(); // clone must occur invariably due to need to store in hashmap

                                    // if not nesterov gt = bt
                                    theta.set(&theta.sub(&(&bt * self.params.lr)?)?)?;
                                    // println!("Momentum {}", bt);
                                    var.b = Some(Var::from_tensor(&bt)?);
                                };
                            }
                        }
                    }
                }
                Momentum::Nesterov(momentum) => {
                    if let Some(decay) = self.params.weight_decay {
                        match decay {
                            Decay::WeightDecay(decay) => {
                                for var in &mut self.vars {
                                    let theta = &var.theta;
                                    // let prev_step = var.b;
                                    if let Some(grad) = grads.get(theta) {
                                        let grad = &(grad + (decay * theta.as_tensor())?)?;
                                        if let Some(prev_step) = &(var.b) {
                                            // println!("Exists");
                                            // bt​←μbt−1​+(1−τ)gt
                                            let bt = ((prev_step.as_tensor() * momentum)?
                                                + (1. - self.params.dampening) * (grad))?;

                                            let gt = (grad + (momentum * &bt)?)?;
                                            // println!("Momentum {}", bt);
                                            prev_step.set(&bt)?;
                                            theta.set(&theta.sub(&(gt * self.params.lr)?)?)?;
                                        } else {
                                            // println!("Doesn't Exist");
                                            // bt​←μbt−1​+(1−τ)gt
                                            // if there is no history bt = gt = grad with no weight_decay
                                            let bt = grad.clone(); // clone must occur invariably due to need to store in hashmap

                                            let gt = (grad + (momentum * &bt)?)?;
                                            // println!("Momentum {}", bt);
                                            var.b = Some(Var::from_tensor(&bt)?);
                                            theta.set(&theta.sub(&(gt * self.params.lr)?)?)?;
                                        };
                                    }
                                }
                            }
                            Decay::DecoupledWeightDecay(decay) => {
                                for var in &mut self.vars {
                                    let theta = &var.theta;
                                    // let prev_step = var.b;
                                    if let Some(grad) = grads.get(theta) {
                                        // decoupled weight decay step
                                        theta.set(
                                            &(theta.as_tensor()
                                                * self.params.lr.mul_add(-decay, 1.))?,
                                        )?;
                                        if let Some(prev_step) = &(var.b) {
                                            // println!("Exists");
                                            // bt​←μbt−1​+(1−τ)gt
                                            let bt = ((prev_step.as_tensor() * momentum)?
                                                + (1. - self.params.dampening) * (grad))?;

                                            let gt = (grad + (momentum * &bt)?)?;
                                            // println!("Momentum {}", bt);
                                            prev_step.set(&bt)?;
                                            theta.set(&theta.sub(&(gt * self.params.lr)?)?)?;
                                        } else {
                                            // println!("Doesn't Exist");
                                            // bt​←μbt−1​+(1−τ)gt
                                            // if there is no history bt = gt = grad with no weight_decay
                                            let bt = grad.clone(); // clone must occur invariably due to need to store in hashmap

                                            let gt = (grad + (momentum * &bt)?)?;
                                            // println!("Momentum {}", bt);
                                            var.b = Some(Var::from_tensor(&bt)?);
                                            theta.set(&theta.sub(&(gt * self.params.lr)?)?)?;
                                        };
                                    }
                                }
                            }
                        }
                    } else {
                        for var in &mut self.vars {
                            let theta = &var.theta;
                            // let prev_step = var.b;
                            if let Some(grad) = grads.get(theta) {
                                if let Some(prev_step) = &(var.b) {
                                    // println!("Exists");
                                    // bt​←μbt−1​+(1−τ)gt
                                    let bt = ((prev_step.as_tensor() * momentum)?
                                        + (1. - self.params.dampening) * (grad))?;

                                    let gt = (grad + (momentum * &bt)?)?;
                                    // println!("Momentum {}", bt);
                                    prev_step.set(&bt)?;
                                    theta.set(&theta.sub(&(gt * self.params.lr)?)?)?;
                                } else {
                                    // println!("Doesn't Exist");
                                    // bt​←μbt−1​+(1−τ)gt
                                    // if there is no history bt = gt = grad with no weight_decay
                                    let bt = grad.clone(); // clone must occur invariably due to need to store in hashmap

                                    let gt = (grad + (momentum * &bt)?)?;
                                    // println!("Momentum {}", bt);
                                    var.b = Some(Var::from_tensor(&bt)?);
                                    theta.set(&theta.sub(&(gt * self.params.lr)?)?)?;
                                };
                            }
                        }
                    }
                }
            }
        } else if let Some(decay) = self.params.weight_decay {
            // These should be the same up to numeric precision
            // For SGD with no momentum decoupled weight decay and L2 reg are equivalent
            match decay {
                Decay::WeightDecay(decay) => {
                    for var in &mut self.vars {
                        let theta = &var.theta;
                        // let prev_step = var.b;
                        if let Some(grad) = grads.get(theta) {
                            let grad = &(grad + (decay * theta.as_tensor())?)?; // weight decay grad
                            theta.set(&theta.sub(&(grad * self.params.lr)?)?)?; // update theta
                        }
                    }
                }
                Decay::DecoupledWeightDecay(decay) => {
                    for var in &mut self.vars {
                        let theta = &var.theta;
                        // let prev_step = var.b;
                        if let Some(grad) = grads.get(theta) {
                            theta
                                .set(&(theta.as_tensor() * self.params.lr.mul_add(-decay, 1.))?)?;
                            theta.set(&theta.sub(&(grad * self.params.lr)?)?)?; // update theta based on grad
                        }
                    }
                }
            }
        } else {
            for var in &mut self.vars {
                let theta = &var.theta;
                // let prev_step = var.b;
                if let Some(grad) = grads.get(theta) {
                    theta.set(&theta.sub(&(grad * self.params.lr)?)?)?; // update theta based on grad
                }
            }
        }

        Ok(())
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }
}

impl OptimParams for SGD {
    fn params(&self) -> &Self::Config {
        &self.params
    }

    fn set_params(&mut self, config: Self::Config) {
        self.params = config;
    }
}

impl SGD {
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
        let params = ParamsSGD {
            lr: 0.004,
            ..Default::default()
        };
        // Now use backprop to run a linear regression between samples and get the coefficients back.
        let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
        let b = Var::new(0f32, &Device::Cpu)?;
        let mut optim = SGD::new(vec![w.clone(), b.clone()], params)?;
        assert_approx_eq!(0.004, optim.learning_rate());
        optim.set_learning_rate(0.002);
        assert_approx_eq!(0.002, optim.learning_rate());
        Ok(())
    }

    #[test]
    fn into_inner_test() -> Result<()> {
        let params = ParamsSGD::default();
        let w = Var::new(&[[3f32, 1.]], &Device::Cpu)?;
        let b = Var::new(-2f32, &Device::Cpu)?;
        let optim = SGD::new(vec![w.clone(), b.clone()], params)?;
        let inner = optim.into_inner();
        assert_eq!(inner[0].as_tensor().to_vec2::<f32>()?, &[[3f32, 1.]]);
        assert_approx_eq!(inner[1].as_tensor().to_vec0::<f32>()?, -2_f32);
        Ok(())
    }
}
