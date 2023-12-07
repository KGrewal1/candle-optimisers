//! The RMS prop algorithm
//!
//! Described in <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>
//!
//! For pseudocde see <https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop>

/*!
RMS prop algorithm

Described in <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>

Pseudocode:

$$
\\begin{aligned}
            &\\rule{110mm}{0.4pt}                                                                 \\\\
            &\\textbf{input}      : \\alpha \\text{ (alpha)},\\: \\gamma \\text{ (lr)},
                \\: \\theta_0 \\text{ (params)}, \\: f(\\theta) \\text{ (objective)}                   \\\\
            &\\hspace{13mm}   \\lambda \\text{ (weight decay)},\\: \\mu \\text{ (momentum)} \\\\
            &\\textbf{initialize} : v_0 \\leftarrow 0 \\text{ (square average)}, \\:
                b_0 \\leftarrow 0 \\text{ (buffer)}, \\: g_0^{ave} \\leftarrow 0     \\\\[-1.ex]
            &\\rule{110mm}{0.4pt}                                                                 \\\\
            &\\textbf{for} \\: t=1 \\: \\textbf{to} \\: \\ldots \\: \\textbf{do}                         \\\\
            &\\hspace{5mm}g_t           \\leftarrow   \\nabla_{\\theta} f_t (\\theta_{t-1})           \\\\
            &\\hspace{5mm}\\textbf{if} \\: \\lambda \\textbf{ is } \\text{Some}                        \\\\
            &\\hspace{10mm} g_t \\leftarrow g_t + \\lambda  \\theta_{t-1}                            \\\\
            &\\hspace{5mm}v_t           \\leftarrow   \\alpha v_{t-1} + (1 - \\alpha) g^2_t
                \\hspace{8mm}                                                                     \\\\
            &\\hspace{5mm} \\tilde{v_t} \\leftarrow v_t                                             \\\\
            &\\hspace{5mm}\\textbf{if} \\: centered                                                          \\\\
            &\\hspace{10mm} g_t^{ave} \\leftarrow g_{t-1}^{ave} \\alpha + (1-\\alpha) g_t            \\\\
            &\\hspace{10mm} \\tilde{v_t} \\leftarrow \\tilde{v_t} -  \\big(g_{t}^{ave} \\big)^2        \\\\
            &\\hspace{5mm}\\textbf{if} \\: \\mu \\textbf{ is } \\text{Some}                                                               \\\\
            &\\hspace{10mm} b_t\\leftarrow \\mu b_{t-1} +
                g_t/ \\big(\\sqrt{\\tilde{v_t}} +  \\epsilon \\big)                                   \\\\
            &\\hspace{10mm} \\theta_t \\leftarrow \\theta_{t-1} - \\gamma b_t                \\\\
            &\\hspace{5mm} \\textbf{else}                                                                  \\\\
            &\\hspace{10mm}\\theta_t      \\leftarrow   \\theta_{t-1} -
                \\gamma  g_t/ \\big(\\sqrt{\\tilde{v_t}} + \\epsilon \\big)  \\hspace{3mm}              \\\\
            &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]
            &\\bf{return} \\:  \\theta_t                                                     \\\\[-1.ex]
            &\\rule{110mm}{0.4pt}                                                          \\\\[-1.ex]
       \\end{aligned}
$$


*/

use candle_core::{Result, Var};
use candle_nn::optim::Optimizer;

/// RMS Prop optimiser
///
/// Described in <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>
///
/// For pseudocde see <https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop>

#[derive(Debug)]
pub struct RMSprop {
    vars: VarRMS,
    params: ParamsRMSprop,
}

trait RmsInner {
    // fn new(vars: Vec<Var>) -> Result<Self>
    // where
    //     Self: Sized;
    fn into_inner(self) -> Vec<Var>;
    fn inner_step(
        &self,
        params: &ParamsRMSprop,
        grads: &candle_core::backprop::GradStore,
    ) -> Result<()>;
}

#[derive(Debug)]
struct VarRMSProp {
    theta: Var,
    v: Var,
}

#[derive(Debug)]
struct VecRMSProp(Vec<VarRMSProp>);

impl VecRMSProp {
    fn new(vars: Vec<Var>) -> Result<Self> {
        Ok(Self(
            vars.into_iter()
                .filter(|var| var.dtype().is_float())
                .map(|var| {
                    let dtype = var.dtype();
                    let shape = var.shape();
                    let device = var.device();
                    let v = Var::zeros(shape, dtype, device)?;
                    Ok(VarRMSProp { theta: var, v })
                })
                .collect::<Result<Vec<VarRMSProp>>>()?,
        ))
    }
}
impl RmsInner for VecRMSProp {
    fn into_inner(self) -> Vec<Var> {
        self.0.into_iter().map(|var| var.theta).collect()
    }

    fn inner_step(
        &self,
        params: &ParamsRMSprop,
        grads: &candle_core::backprop::GradStore,
    ) -> Result<()> {
        if let Some(wd) = params.weight_decay {
            for var in &self.0 {
                let theta = &var.theta;
                let v = &var.v;
                if let Some(grad) = grads.get(theta) {
                    let grad = &(grad + (wd * theta.as_tensor())?)?;
                    let v_next = ((params.alpha * v.as_tensor())?
                        + ((1. - params.alpha) * grad.powf(2.)?)?)?;

                    let delta = (grad / (v_next.sqrt()? + params.eps)?)?;
                    theta.set(&theta.sub(&(params.lr * delta)?)?)?;
                    v.set(&v_next)?;
                }
            }
        } else {
            for var in &self.0 {
                let theta = &var.theta;
                let v = &var.v;
                if let Some(grad) = grads.get(theta) {
                    let v_next = ((params.alpha * v.as_tensor())?
                        + ((1. - params.alpha) * grad.powf(2.)?)?)?;

                    let delta = (grad / (v_next.sqrt()? + params.eps)?)?;
                    theta.set(&theta.sub(&(params.lr * delta)?)?)?;
                    v.set(&v_next)?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct VarRMSPropCentered {
    theta: Var,
    v: Var,
    g: Var,
}

#[derive(Debug)]
struct VecRmsPropCentered(Vec<VarRMSPropCentered>);

impl VecRmsPropCentered {
    fn new(vars: Vec<Var>) -> Result<Self> {
        Ok(Self(
            vars.into_iter()
                .filter(|var| var.dtype().is_float())
                .map(|var| {
                    let dtype = var.dtype();
                    let shape = var.shape();
                    let device = var.device();
                    let v = Var::zeros(shape, dtype, device)?;
                    let g = Var::zeros(shape, dtype, device)?;
                    Ok(VarRMSPropCentered { theta: var, v, g })
                })
                .collect::<Result<Vec<VarRMSPropCentered>>>()?,
        ))
    }
}

impl RmsInner for VecRmsPropCentered {
    fn into_inner(self) -> Vec<Var> {
        self.0.into_iter().map(|var| var.theta).collect()
    }

    fn inner_step(
        &self,
        params: &ParamsRMSprop,
        grads: &candle_core::backprop::GradStore,
    ) -> Result<()> {
        if let Some(wd) = params.weight_decay {
            for var in &self.0 {
                let theta = &var.theta;
                let v = &var.v;
                let g_avg = &var.g;
                if let Some(grad) = grads.get(theta) {
                    let grad = &(grad + (wd * theta.as_tensor())?)?;
                    let v_next = ((params.alpha * v.as_tensor())?
                        + ((1. - params.alpha) * grad.powf(2.)?)?)?;
                    let (v_tilde, g_next) = {
                        let g =
                            ((params.alpha * g_avg.as_tensor())? + ((1. - params.alpha) * grad)?)?;
                        ((&v_next - g.powf(2.)?)?, g)
                    };

                    let delta = (grad / (v_tilde.sqrt()? + params.eps)?)?;
                    theta.set(&theta.sub(&(params.lr * delta)?)?)?;
                    v.set(&v_next)?;
                    g_avg.set(&g_next)?;
                }
            }
        } else {
            for var in &self.0 {
                let theta = &var.theta;
                let v = &var.v;
                let g_avg = &var.g;
                if let Some(grad) = grads.get(theta) {
                    let v_next = ((params.alpha * v.as_tensor())?
                        + ((1. - params.alpha) * grad.powf(2.)?)?)?;
                    let (v_tilde, g_next) = {
                        let g =
                            ((params.alpha * g_avg.as_tensor())? + ((1. - params.alpha) * grad)?)?;
                        ((&v_next - g.powf(2.)?)?, g)
                    };

                    let delta = (grad / (v_tilde.sqrt()? + params.eps)?)?;
                    theta.set(&theta.sub(&(params.lr * delta)?)?)?;
                    v.set(&v_next)?;
                    g_avg.set(&g_next)?;
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
struct VarRMSPropMomentum {
    theta: Var,
    v: Var,
    b: Var,
}

#[derive(Debug)]
struct VecRmsPropMomentum {
    vars: Vec<VarRMSPropMomentum>,
    momentum: f64,
}

impl VecRmsPropMomentum {
    fn new(vars: Vec<Var>, momentum: f64) -> Result<Self> {
        Ok(Self {
            vars: vars
                .into_iter()
                .filter(|var| var.dtype().is_float())
                .map(|var| {
                    let dtype = var.dtype();
                    let shape = var.shape();
                    let device = var.device();
                    let v = Var::zeros(shape, dtype, device)?;
                    let b = Var::zeros(shape, dtype, device)?;
                    Ok(VarRMSPropMomentum { theta: var, v, b })
                })
                .collect::<Result<Vec<VarRMSPropMomentum>>>()?,
            momentum,
        })
    }
}

impl RmsInner for VecRmsPropMomentum {
    fn into_inner(self) -> Vec<Var> {
        self.vars.into_iter().map(|var| var.theta).collect()
    }

    fn inner_step(
        &self,
        params: &ParamsRMSprop,
        grads: &candle_core::backprop::GradStore,
    ) -> Result<()> {
        if let Some(wd) = params.weight_decay {
            for var in &self.vars {
                let theta = &var.theta;
                let v = &var.v;
                let b = &var.b;
                if let Some(grad) = grads.get(theta) {
                    let grad = &(grad + (wd * theta.as_tensor())?)?;
                    let v_next = ((params.alpha * v.as_tensor())?
                        + ((1. - params.alpha) * grad.powf(2.)?)?)?;

                    let b_next = ((self.momentum * b.as_tensor())?
                        + (grad / (v_next.sqrt()? + params.eps)?)?)?;
                    theta.set(&theta.sub(&(params.lr * &b_next)?)?)?;
                    v.set(&v_next)?;
                    b.set(&b_next)?;
                }
            }
        } else {
            for var in &self.vars {
                let theta = &var.theta;
                let v = &var.v;
                let b = &var.b;
                if let Some(grad) = grads.get(theta) {
                    let v_next = ((params.alpha * v.as_tensor())?
                        + ((1. - params.alpha) * grad.powf(2.)?)?)?;

                    let b_next = ((self.momentum * b.as_tensor())?
                        + (grad / (v_next.sqrt()? + params.eps)?)?)?;
                    theta.set(&theta.sub(&(params.lr * &b_next)?)?)?;
                    v.set(&v_next)?;
                    b.set(&b_next)?;
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
struct VarRMSPropMomentumCentered {
    theta: Var,
    v: Var,
    g: Var,
    b: Var,
}

#[derive(Debug)]
struct VecRmsPropMomentumCentered {
    vars: Vec<VarRMSPropMomentumCentered>,
    momentum: f64,
}

impl VecRmsPropMomentumCentered {
    fn new(vars: Vec<Var>, momentum: f64) -> Result<Self> {
        Ok(Self {
            vars: vars
                .into_iter()
                .filter(|var| var.dtype().is_float())
                .map(|var| {
                    let dtype = var.dtype();
                    let shape = var.shape();
                    let device = var.device();
                    let v = Var::zeros(shape, dtype, device)?;
                    let b = Var::zeros(shape, dtype, device)?;
                    let g = Var::zeros(shape, dtype, device)?;
                    Ok(VarRMSPropMomentumCentered {
                        theta: var,
                        v,
                        b,
                        g,
                    })
                })
                .collect::<Result<Vec<VarRMSPropMomentumCentered>>>()?,
            momentum,
        })
    }
}

impl RmsInner for VecRmsPropMomentumCentered {
    fn into_inner(self) -> Vec<Var> {
        self.vars.into_iter().map(|var| var.theta).collect()
    }

    fn inner_step(
        &self,
        params: &ParamsRMSprop,
        grads: &candle_core::backprop::GradStore,
    ) -> Result<()> {
        if let Some(wd) = params.weight_decay {
            for var in &self.vars {
                let theta = &var.theta;
                let v = &var.v;
                let g_avg = &var.g;
                let b = &var.b;
                if let Some(grad) = grads.get(theta) {
                    let grad = &(grad + (wd * theta.as_tensor())?)?;
                    let v_next = ((params.alpha * v.as_tensor())?
                        + ((1. - params.alpha) * grad.powf(2.)?)?)?;

                    let (v_tilde, g_next) = {
                        let g =
                            ((params.alpha * g_avg.as_tensor())? + ((1. - params.alpha) * grad)?)?;
                        ((&v_next - g.powf(2.)?)?, g)
                    };

                    let b_next = ((self.momentum * b.as_tensor())?
                        + (grad / (v_tilde.sqrt()? + params.eps)?)?)?;
                    theta.set(&theta.sub(&(params.lr * &b_next)?)?)?;
                    v.set(&v_next)?;
                    g_avg.set(&g_next)?;
                    b.set(&b_next)?;
                }
            }
        } else {
            for var in &self.vars {
                let theta = &var.theta;
                let v = &var.v;
                let g_avg = &var.g;
                let b = &var.b;
                if let Some(grad) = grads.get(theta) {
                    let v_next = ((params.alpha * v.as_tensor())?
                        + ((1. - params.alpha) * grad.powf(2.)?)?)?;

                    let (v_tilde, g_next) = {
                        let g =
                            ((params.alpha * g_avg.as_tensor())? + ((1. - params.alpha) * grad)?)?;
                        ((&v_next - g.powf(2.)?)?, g)
                    };

                    let b_next = ((self.momentum * b.as_tensor())?
                        + (grad / (v_tilde.sqrt()? + params.eps)?)?)?;
                    theta.set(&theta.sub(&(params.lr * &b_next)?)?)?;
                    v.set(&v_next)?;
                    g_avg.set(&g_next)?;
                    b.set(&b_next)?;
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
enum VarRMS {
    RMSProp(VecRMSProp),
    Centered(VecRmsPropCentered),
    Momentum(VecRmsPropMomentum),
    MomentumCentered(VecRmsPropMomentumCentered),
}

/// Parameters for RMSprop
#[derive(Debug)]
pub struct ParamsRMSprop {
    /// Learning rate
    pub lr: f64,
    /// Smoothing constant
    pub alpha: f64,
    /// Term added to the denominator to improve numerical stability
    pub eps: f64,
    /// Weight decay
    pub weight_decay: Option<f64>,
    /// Momentum
    pub momentum: Option<f64>,
    /// Whether to use centered RMSprop, normalising the gradient by an estimate of its variance
    pub centered: bool,
}

impl Default for ParamsRMSprop {
    fn default() -> Self {
        Self {
            lr: 0.01,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: None,
            momentum: None,
            centered: false,
        }
    }
}

impl Optimizer for RMSprop {
    type Config = ParamsRMSprop;

    fn new(vars: Vec<Var>, params: ParamsRMSprop) -> Result<Self> {
        if let Some(momentum) = params.momentum {
            if params.centered {
                Ok(Self {
                    vars: VarRMS::MomentumCentered(VecRmsPropMomentumCentered::new(
                        vars, momentum,
                    )?),
                    params,
                })
            } else {
                Ok(Self {
                    vars: VarRMS::Momentum(VecRmsPropMomentum::new(vars, momentum)?),
                    params,
                })
            }
        } else if params.centered {
            // no need to have b tensor
            // need to have g tensor
            Ok(Self {
                vars: VarRMS::Centered(VecRmsPropCentered::new(vars)?),
                params,
            })
        } else {
            // no need to have b tensor
            // no need to have g tensor
            // let vars = vars
            //     .into_iter()
            //     .filter(|var| var.dtype().is_float())
            //     .map(|var| {
            //         let dtype = var.dtype();
            //         let shape = var.shape();
            //         let device = var.device();
            //         let v = Var::zeros(shape, dtype, device)?;
            //         Ok(VarRMS::RMSProp(VarRMSProp { theta: var, v }))
            //     })
            //     .collect::<Result<Vec<VarRMS>>>()?;

            Ok(Self {
                vars: VarRMS::RMSProp(VecRMSProp::new(vars)?),
                params,
            })
        }
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        match &self.vars {
            VarRMS::RMSProp(vars) => vars.inner_step(&self.params, grads),
            VarRMS::Centered(vars) => vars.inner_step(&self.params, grads),
            VarRMS::Momentum(vars) => vars.inner_step(&self.params, grads),
            VarRMS::MomentumCentered(vars) => vars.inner_step(&self.params, grads),
        }
        // Ok(())
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }
}

impl RMSprop {
    #[must_use]
    pub fn into_inner(self) -> Vec<Var> {
        match self.vars {
            VarRMS::RMSProp(vars) => vars.into_inner(),
            VarRMS::Centered(vars) => vars.into_inner(),
            VarRMS::Momentum(vars) => vars.into_inner(),
            VarRMS::MomentumCentered(vars) => vars.into_inner(),
        }
    }
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

    #[test]
    fn into_inner_test() -> Result<()> {
        let params = ParamsRMSprop::default();
        let w = Var::new(&[[3f32, 1.]], &Device::Cpu)?;
        let b = Var::new(-2f32, &Device::Cpu)?;
        let n_sgd = RMSprop::new(vec![w.clone(), b.clone()], params)?;
        let inner = n_sgd.into_inner();
        assert_eq!(inner[0].as_tensor().to_vec2::<f32>()?, &[[3f32, 1.]]);
        assert_approx_eq!(inner[1].as_tensor().to_vec0::<f32>()?, -2_f32);

        let params = ParamsRMSprop {
            centered: true,
            ..Default::default()
        };
        let w = Var::new(&[[3f32, 1.]], &Device::Cpu)?;
        let b = Var::new(-2f32, &Device::Cpu)?;
        let n_sgd = RMSprop::new(vec![w.clone(), b.clone()], params)?;
        let inner = n_sgd.into_inner();
        assert_eq!(inner[0].as_tensor().to_vec2::<f32>()?, &[[3f32, 1.]]);
        assert_approx_eq!(inner[1].as_tensor().to_vec0::<f32>()?, -2_f32);

        let params = ParamsRMSprop {
            momentum: Some(0.004),
            ..Default::default()
        };
        let w = Var::new(&[[3f32, 1.]], &Device::Cpu)?;
        let b = Var::new(-2f32, &Device::Cpu)?;
        let n_sgd = RMSprop::new(vec![w.clone(), b.clone()], params)?;
        let inner = n_sgd.into_inner();
        assert_eq!(inner[0].as_tensor().to_vec2::<f32>()?, &[[3f32, 1.]]);
        assert_approx_eq!(inner[1].as_tensor().to_vec0::<f32>()?, -2_f32);

        let params = ParamsRMSprop {
            centered: true,
            momentum: Some(0.004),
            ..Default::default()
        };
        let b = Var::new(-2f32, &Device::Cpu)?;
        let n_sgd = RMSprop::new(vec![w.clone(), b.clone()], params)?;
        let inner = n_sgd.into_inner();
        assert_eq!(inner[0].as_tensor().to_vec2::<f32>()?, &[[3f32, 1.]]);
        assert_approx_eq!(inner[1].as_tensor().to_vec0::<f32>()?, -2_f32);
        Ok(())
    }
}
