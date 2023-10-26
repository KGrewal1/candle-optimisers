use candle_core::{Result, Var};
use candle_nn::optim::Optimizer;

/// RMS Prop optimiser
///
/// Described in <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>
///
/// For pseudocde see <https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop>

#[derive(Debug)]
pub struct RMSprop {
    vars: Vec<VarRMS>,
    params: ParamsRMSprop,
}

#[derive(Debug)]
struct VarRMSProp {
    theta: Var,
    v: Var,
}

#[derive(Debug)]
struct VarRMSPropCentered {
    theta: Var,
    v: Var,
    g: Var,
}

#[derive(Debug)]
struct VarRMSPropMomentum {
    theta: Var,
    v: Var,
    b: Var,
}

#[derive(Debug)]
struct VarRMSPropMomentumCentered {
    theta: Var,
    v: Var,
    g: Var,
    b: Var,
}

#[derive(Debug)]
enum VarRMS {
    RMSProp(VarRMSProp),
    Centered(VarRMSPropCentered),
    Momentum(VarRMSPropMomentum),
    MomentumCentered(VarRMSPropMomentumCentered),
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
        if params.momentum == 0. && params.centered {
            // no need to have b tensor
            // need to have g tensor
            let vars = vars
                .into_iter()
                .filter(|var| var.dtype().is_float())
                .map(|var| {
                    let dtype = var.dtype();
                    let shape = var.shape();
                    let device = var.device();
                    let v = Var::zeros(shape, dtype, device)?;
                    let g = Var::zeros(shape, dtype, device)?;
                    Ok(VarRMS::Centered(VarRMSPropCentered { theta: var, v, g }))
                })
                .collect::<Result<Vec<VarRMS>>>()?;

            Ok(Self { vars, params })
        } else if params.momentum == 0. && !params.centered {
            // no need to have b tensor
            // no need to have g tensor
            let vars = vars
                .into_iter()
                .filter(|var| var.dtype().is_float())
                .map(|var| {
                    let dtype = var.dtype();
                    let shape = var.shape();
                    let device = var.device();
                    let v = Var::zeros(shape, dtype, device)?;
                    Ok(VarRMS::RMSProp(VarRMSProp { theta: var, v }))
                })
                .collect::<Result<Vec<VarRMS>>>()?;

            Ok(Self { vars, params })
        } else if params.momentum != 0. && !params.centered {
            // need to have b tensor
            // no need to have g tensor
            let vars = vars
                .into_iter()
                .filter(|var| var.dtype().is_float())
                .map(|var| {
                    let dtype = var.dtype();
                    let shape = var.shape();
                    let device = var.device();
                    let v = Var::zeros(shape, dtype, device)?;
                    let b = Var::zeros(shape, dtype, device)?;
                    Ok(VarRMS::Momentum(VarRMSPropMomentum { theta: var, v, b }))
                })
                .collect::<Result<Vec<VarRMS>>>()?;

            Ok(Self { vars, params })
        } else {
            // need to have b tensor
            // need to have g tensor
            let vars = vars
                .into_iter()
                .filter(|var| var.dtype().is_float())
                .map(|var| {
                    let dtype = var.dtype();
                    let shape = var.shape();
                    let device = var.device();
                    let v = Var::zeros(shape, dtype, device)?;
                    let b = Var::zeros(shape, dtype, device)?;
                    let g = Var::zeros(shape, dtype, device)?;
                    Ok(VarRMS::MomentumCentered(VarRMSPropMomentumCentered {
                        theta: var,
                        v,
                        b,
                        g,
                    }))
                })
                .collect::<Result<Vec<VarRMS>>>()?;

            Ok(Self { vars, params })
        }
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    #[allow(clippy::too_many_lines)]
    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        for var in &self.vars {
            match var {
                VarRMS::RMSProp(var) => {
                    let theta = &var.theta;
                    let v = &var.v;
                    if let Some(grad) = grads.get(theta) {
                        if self.params.weight_decay == 0. {
                            let v_next = ((self.params.alpha * v.as_tensor())?
                                + ((1. - self.params.alpha) * grad.powf(2.)?)?)?;

                            let delta = (grad / (v_next.sqrt()? + self.params.eps)?)?;
                            theta.set(&theta.sub(&(self.params.lr * delta)?)?)?;
                            v.set(&v_next)?;
                        } else {
                            let grad = &(grad + (self.params.weight_decay * theta.as_tensor())?)?;
                            let v_next = ((self.params.alpha * v.as_tensor())?
                                + ((1. - self.params.alpha) * grad.powf(2.)?)?)?;

                            let delta = (grad / (v_next.sqrt()? + self.params.eps)?)?;
                            theta.set(&theta.sub(&(self.params.lr * delta)?)?)?;
                            v.set(&v_next)?;
                        }
                    }
                }
                VarRMS::Centered(var) => {
                    let theta = &var.theta;
                    let v = &var.v;
                    let g_avg = &var.g;
                    if let Some(grad) = grads.get(theta) {
                        if self.params.weight_decay == 0. {
                            let v_next = ((self.params.alpha * v.as_tensor())?
                                + ((1. - self.params.alpha) * grad.powf(2.)?)?)?;
                            let (v_tilde, g_next) = {
                                let g = ((self.params.alpha * g_avg.as_tensor())?
                                    + ((1. - self.params.alpha) * grad)?)?;
                                ((&v_next - g.powf(2.)?)?, g)
                            };

                            let delta = (grad / (v_tilde.sqrt()? + self.params.eps)?)?;
                            theta.set(&theta.sub(&(self.params.lr * delta)?)?)?;
                            v.set(&v_next)?;
                            g_avg.set(&g_next)?;
                        } else {
                            let grad = &(grad + (self.params.weight_decay * theta.as_tensor())?)?;
                            let v_next = ((self.params.alpha * v.as_tensor())?
                                + ((1. - self.params.alpha) * grad.powf(2.)?)?)?;
                            let (v_tilde, g_next) = {
                                let g = ((self.params.alpha * g_avg.as_tensor())?
                                    + ((1. - self.params.alpha) * grad)?)?;
                                ((&v_next - g.powf(2.)?)?, g)
                            };

                            let delta = (grad / (v_tilde.sqrt()? + self.params.eps)?)?;
                            theta.set(&theta.sub(&(self.params.lr * delta)?)?)?;
                            v.set(&v_next)?;
                            g_avg.set(&g_next)?;
                        }
                    }
                }
                VarRMS::Momentum(var) => {
                    let theta = &var.theta;
                    let v = &var.v;
                    let b = &var.b;
                    if let Some(grad) = grads.get(theta) {
                        if self.params.weight_decay == 0. {
                            let v_next = ((self.params.alpha * v.as_tensor())?
                                + ((1. - self.params.alpha) * grad.powf(2.)?)?)?;

                            let b_next = ((self.params.momentum * b.as_tensor())?
                                + (grad / (v_next.sqrt()? + self.params.eps)?)?)?;
                            theta.set(&theta.sub(&(self.params.lr * &b_next)?)?)?;
                            v.set(&v_next)?;
                            b.set(&b_next)?;
                        } else {
                            let grad = &(grad + (self.params.weight_decay * theta.as_tensor())?)?;
                            let v_next = ((self.params.alpha * v.as_tensor())?
                                + ((1. - self.params.alpha) * grad.powf(2.)?)?)?;

                            let b_next = ((self.params.momentum * b.as_tensor())?
                                + (grad / (v_next.sqrt()? + self.params.eps)?)?)?;
                            theta.set(&theta.sub(&(self.params.lr * &b_next)?)?)?;
                            v.set(&v_next)?;
                            b.set(&b_next)?;
                        }
                    }
                }
                VarRMS::MomentumCentered(var) => {
                    let theta = &var.theta;
                    let v = &var.v;
                    let g_avg = &var.g;
                    let b = &var.b;
                    if let Some(grad) = grads.get(theta) {
                        if self.params.weight_decay == 0. {
                            let v_next = ((self.params.alpha * v.as_tensor())?
                                + ((1. - self.params.alpha) * grad.powf(2.)?)?)?;

                            let (v_tilde, g_next) = {
                                let g = ((self.params.alpha * g_avg.as_tensor())?
                                    + ((1. - self.params.alpha) * grad)?)?;
                                ((&v_next - g.powf(2.)?)?, g)
                            };

                            let b_next = ((self.params.momentum * b.as_tensor())?
                                + (grad / (v_tilde.sqrt()? + self.params.eps)?)?)?;
                            theta.set(&theta.sub(&(self.params.lr * &b_next)?)?)?;
                            v.set(&v_next)?;
                            g_avg.set(&g_next)?;
                            b.set(&b_next)?;
                        } else {
                            let grad = &(grad + (self.params.weight_decay * theta.as_tensor())?)?;
                            let v_next = ((self.params.alpha * v.as_tensor())?
                                + ((1. - self.params.alpha) * grad.powf(2.)?)?)?;

                            let (v_tilde, g_next) = {
                                let g = ((self.params.alpha * g_avg.as_tensor())?
                                    + ((1. - self.params.alpha) * grad)?)?;
                                ((&v_next - g.powf(2.)?)?, g)
                            };

                            let b_next = ((self.params.momentum * b.as_tensor())?
                                + (grad / (v_tilde.sqrt()? + self.params.eps)?)?)?;
                            theta.set(&theta.sub(&(self.params.lr * &b_next)?)?)?;
                            v.set(&v_next)?;
                            g_avg.set(&g_next)?;
                            b.set(&b_next)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }
}

// impl RMSprop {
//     #[must_use]
//     pub fn into_inner(self) -> Vec<Var> {
//         self.vars
//     }

//     pub fn push(&mut self, var: &Var) {
//         self.vars.push(var.clone());
//     }
// }

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
