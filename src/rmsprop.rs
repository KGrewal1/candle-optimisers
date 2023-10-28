//! The RMS prop algoirithm

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
    fn new(vars: Vec<Var>) -> Result<Self>
    where
        Self: Sized;
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

impl RmsInner for VecRMSProp {
    fn new(vars: Vec<Var>) -> Result<Self> {
        Ok(VecRMSProp(
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

    fn into_inner(self) -> Vec<Var> {
        self.0.into_iter().map(|var| var.theta).collect()
    }

    fn inner_step(
        &self,
        params: &ParamsRMSprop,
        grads: &candle_core::backprop::GradStore,
    ) -> Result<()> {
        if params.weight_decay == 0. {
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
        } else {
            for var in &self.0 {
                let theta = &var.theta;
                let v = &var.v;
                if let Some(grad) = grads.get(theta) {
                    let grad = &(grad + (params.weight_decay * theta.as_tensor())?)?;
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

impl RmsInner for VecRmsPropCentered {
    fn new(vars: Vec<Var>) -> Result<Self> {
        Ok(VecRmsPropCentered(
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

    fn into_inner(self) -> Vec<Var> {
        self.0.into_iter().map(|var| var.theta).collect()
    }

    fn inner_step(
        &self,
        params: &ParamsRMSprop,
        grads: &candle_core::backprop::GradStore,
    ) -> Result<()> {
        if params.weight_decay == 0. {
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
        } else {
            for var in &self.0 {
                let theta = &var.theta;
                let v = &var.v;
                let g_avg = &var.g;
                if let Some(grad) = grads.get(theta) {
                    let grad = &(grad + (params.weight_decay * theta.as_tensor())?)?;
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
struct VecRmsPropMomentum(Vec<VarRMSPropMomentum>);

impl RmsInner for VecRmsPropMomentum {
    fn new(vars: Vec<Var>) -> Result<Self> {
        Ok(VecRmsPropMomentum(
            vars.into_iter()
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
        ))
    }
    fn into_inner(self) -> Vec<Var> {
        self.0.into_iter().map(|var| var.theta).collect()
    }

    fn inner_step(
        &self,
        params: &ParamsRMSprop,
        grads: &candle_core::backprop::GradStore,
    ) -> Result<()> {
        if params.weight_decay == 0. {
            for var in &self.0 {
                let theta = &var.theta;
                let v = &var.v;
                let b = &var.b;
                if let Some(grad) = grads.get(theta) {
                    let v_next = ((params.alpha * v.as_tensor())?
                        + ((1. - params.alpha) * grad.powf(2.)?)?)?;

                    let b_next = ((params.momentum * b.as_tensor())?
                        + (grad / (v_next.sqrt()? + params.eps)?)?)?;
                    theta.set(&theta.sub(&(params.lr * &b_next)?)?)?;
                    v.set(&v_next)?;
                    b.set(&b_next)?;
                }
            }
        } else {
            for var in &self.0 {
                let theta = &var.theta;
                let v = &var.v;
                let b = &var.b;
                if let Some(grad) = grads.get(theta) {
                    let grad = &(grad + (params.weight_decay * theta.as_tensor())?)?;
                    let v_next = ((params.alpha * v.as_tensor())?
                        + ((1. - params.alpha) * grad.powf(2.)?)?)?;

                    let b_next = ((params.momentum * b.as_tensor())?
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
struct VecRmsPropMomentumCentered(Vec<VarRMSPropMomentumCentered>);

impl RmsInner for VecRmsPropMomentumCentered {
    fn new(vars: Vec<Var>) -> Result<Self> {
        Ok(VecRmsPropMomentumCentered(
            vars.into_iter()
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
        ))
    }

    fn into_inner(self) -> Vec<Var> {
        self.0.into_iter().map(|var| var.theta).collect()
    }

    fn inner_step(
        &self,
        params: &ParamsRMSprop,
        grads: &candle_core::backprop::GradStore,
    ) -> Result<()> {
        if params.weight_decay == 0. {
            for var in &self.0 {
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

                    let b_next = ((params.momentum * b.as_tensor())?
                        + (grad / (v_tilde.sqrt()? + params.eps)?)?)?;
                    theta.set(&theta.sub(&(params.lr * &b_next)?)?)?;
                    v.set(&v_next)?;
                    g_avg.set(&g_next)?;
                    b.set(&b_next)?;
                }
            }
        } else {
            for var in &self.0 {
                let theta = &var.theta;
                let v = &var.v;
                let g_avg = &var.g;
                let b = &var.b;
                if let Some(grad) = grads.get(theta) {
                    let grad = &(grad + (params.weight_decay * theta.as_tensor())?)?;
                    let v_next = ((params.alpha * v.as_tensor())?
                        + ((1. - params.alpha) * grad.powf(2.)?)?)?;

                    let (v_tilde, g_next) = {
                        let g =
                            ((params.alpha * g_avg.as_tensor())? + ((1. - params.alpha) * grad)?)?;
                        ((&v_next - g.powf(2.)?)?, g)
                    };

                    let b_next = ((params.momentum * b.as_tensor())?
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
            Ok(Self {
                vars: VarRMS::Centered(VecRmsPropCentered::new(vars)?),
                params,
            })
        } else if params.momentum == 0. && !params.centered {
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
        } else if params.momentum != 0. && !params.centered {
            // need to have b tensor
            // no need to have g tensor

            Ok(Self {
                vars: VarRMS::Momentum(VecRmsPropMomentum::new(vars)?),
                params,
            })
        } else {
            // need to have b tensor
            // need to have g tensor

            Ok(Self {
                vars: VarRMS::MomentumCentered(VecRmsPropMomentumCentered::new(vars)?),
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
            momentum: 0.004,
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
            momentum: 0.004,
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
