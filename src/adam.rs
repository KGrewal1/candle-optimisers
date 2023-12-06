//! The Adam optimiser

use candle_core::{Result, Var};
use candle_nn::optim::Optimizer;

use crate::Decay;

/// Adam optimiser
///
/// Described in <https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ>
///
/// For pseudocde see <https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam>
/// Note that this is the same as pytorch implementation not pseudocode (hat of vmax not max of vhat)
/// This includes decoupled weight decay (Adam W)

trait AdamInner {
    fn new(vars: Vec<Var>) -> Result<Self>
    where
        Self: Sized;
    fn into_inner(self) -> Vec<Var>;
    fn inner_step(
        &self,
        params: &ParamsAdam,
        grads: &candle_core::backprop::GradStore,
        t: f64,
    ) -> Result<()>;
}

#[derive(Debug)]
pub struct Adam {
    vars: VarAdam,
    params: ParamsAdam,
    t: f64,
}

#[derive(Debug)]
struct VarAdamBase {
    theta: Var,
    m: Var,
    v: Var,
}

#[derive(Debug)]
struct VecAdamBase(Vec<VarAdamBase>);

impl AdamInner for VecAdamBase {
    fn new(vars: Vec<Var>) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(VecAdamBase(
            vars.into_iter()
                .filter(|var| var.dtype().is_float())
                .map(|var| {
                    let dtype = var.dtype();
                    let shape = var.shape();
                    let device = var.device();
                    let m = Var::zeros(shape, dtype, device)?;
                    let v = Var::zeros(shape, dtype, device)?;
                    Ok(VarAdamBase { theta: var, m, v })
                })
                .collect::<Result<Vec<VarAdamBase>>>()?,
        ))
    }

    fn into_inner(self) -> Vec<Var> {
        self.0.into_iter().map(|var| var.theta).collect()
    }

    fn inner_step(
        &self,
        params: &ParamsAdam,
        grads: &candle_core::backprop::GradStore,
        t: f64,
    ) -> Result<()> {
        if let Some(decay) = params.weight_decay {
            match decay {
                Decay::WeightDecay(decay) => {
                    for var in &self.0 {
                        let theta = &var.theta;
                        let m = &var.m;
                        let v = &var.v;
                        if let Some(grad) = grads.get(theta) {
                            let grad = &(grad + (decay * theta.as_tensor())?)?;
                            let m_next = ((params.beta_1 * m.as_tensor())?
                                + ((1. - params.beta_1) * grad)?)?;
                            let v_next = ((params.beta_2 * v.as_tensor())?
                                + ((1. - params.beta_2) * grad.powf(2.)?)?)?;
                            let m_hat = (&m_next / (1. - (params.beta_1).powf(t)))?;
                            let v_hat = (&v_next / (1. - params.beta_2.powf(t)))?;
                            let delta =
                                (m_hat * params.lr)?.div(&(v_hat.powf(0.5)? + params.eps)?)?;
                            theta.set(&theta.sub(&(delta))?)?;
                            m.set(&m_next)?;
                            v.set(&v_next)?;
                        }
                    }
                }
                Decay::DecoupledWeightDecay(decay) => {
                    for var in &self.0 {
                        let theta = &var.theta;
                        let m = &var.m;
                        let v = &var.v;
                        if let Some(grad) = grads.get(theta) {
                            theta.set(&(theta.as_tensor() * params.lr.mul_add(-decay, 1.))?)?;
                            let m_next = ((params.beta_1 * m.as_tensor())?
                                + ((1. - params.beta_1) * grad)?)?;
                            let v_next = ((params.beta_2 * v.as_tensor())?
                                + ((1. - params.beta_2) * grad.powf(2.)?)?)?;
                            let m_hat = (&m_next / (1. - (params.beta_1).powf(t)))?;
                            let v_hat = (&v_next / (1. - params.beta_2.powf(t)))?;
                            let delta =
                                (m_hat * params.lr)?.div(&(v_hat.powf(0.5)? + params.eps)?)?;
                            theta.set(&theta.sub(&(delta))?)?;
                            m.set(&m_next)?;
                            v.set(&v_next)?;
                        }
                    }
                }
            }
        } else {
            for var in &self.0 {
                let theta = &var.theta;
                let m = &var.m;
                let v = &var.v;
                if let Some(grad) = grads.get(theta) {
                    let m_next =
                        ((params.beta_1 * m.as_tensor())? + ((1. - params.beta_1) * grad)?)?;
                    let v_next = ((params.beta_2 * v.as_tensor())?
                        + ((1. - params.beta_2) * grad.powf(2.)?)?)?;
                    let m_hat = (&m_next / (1. - (params.beta_1).powf(t)))?;
                    let v_hat = (&v_next / (1. - params.beta_2.powf(t)))?;
                    let delta = (m_hat * params.lr)?.div(&(v_hat.powf(0.5)? + params.eps)?)?;
                    theta.set(&theta.sub(&(delta))?)?;
                    m.set(&m_next)?;
                    v.set(&v_next)?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct VarAdamAmsgrad {
    theta: Var,
    m: Var,
    v: Var,
    vmax: Var,
}

#[derive(Debug)]
struct VecAdamAmsgrad(Vec<VarAdamAmsgrad>);

impl AdamInner for VecAdamAmsgrad {
    fn new(vars: Vec<Var>) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(VecAdamAmsgrad(
            vars.into_iter()
                .filter(|var| var.dtype().is_float())
                .map(|var| {
                    let dtype = var.dtype();
                    let shape = var.shape();
                    let device = var.device();
                    let m = Var::zeros(shape, dtype, device)?;
                    let v = Var::zeros(shape, dtype, device)?;
                    let vmax = Var::zeros(shape, dtype, device)?;
                    Ok(VarAdamAmsgrad {
                        theta: var,
                        m,
                        v,
                        vmax,
                    })
                })
                .collect::<Result<Vec<VarAdamAmsgrad>>>()?,
        ))
    }

    fn into_inner(self) -> Vec<Var> {
        self.0.into_iter().map(|var| var.theta).collect()
    }

    fn inner_step(
        &self,
        params: &ParamsAdam,
        grads: &candle_core::backprop::GradStore,
        t: f64,
    ) -> Result<()> {
        if let Some(decay) = params.weight_decay {
            match decay {
                Decay::WeightDecay(decay) => {
                    for var in &self.0 {
                        let theta = &var.theta;
                        let m = &var.m;
                        let v = &var.v;
                        let vmax = &var.vmax;
                        if let Some(grad) = grads.get(theta) {
                            let grad = &(grad + (decay * theta.as_tensor())?)?;
                            let m_next = ((params.beta_1 * m.as_tensor())?
                                + ((1. - params.beta_1) * grad)?)?;
                            let v_next = ((params.beta_2 * v.as_tensor())?
                                + ((1. - params.beta_2) * grad.powf(2.)?)?)?;
                            let m_hat = (&m_next / (1. - (params.beta_1).powf(t)))?;
                            let vmax_next = vmax.maximum(&v_next)?;
                            let v_hat = (&vmax_next / (1. - params.beta_2.powf(t)))?;
                            let delta =
                                (m_hat * params.lr)?.div(&(v_hat.powf(0.5)? + params.eps)?)?;
                            theta.set(&theta.sub(&(delta))?)?;
                            m.set(&m_next)?;
                            v.set(&v_next)?;
                            vmax.set(&vmax_next)?;
                        }
                    }
                }
                Decay::DecoupledWeightDecay(decay) => {
                    for var in &self.0 {
                        let theta = &var.theta;
                        let m = &var.m;
                        let v = &var.v;
                        let vmax = &var.vmax;
                        if let Some(grad) = grads.get(theta) {
                            theta.set(&(theta.as_tensor() * params.lr.mul_add(-decay, 1.))?)?;
                            let m_next = ((params.beta_1 * m.as_tensor())?
                                + ((1. - params.beta_1) * grad)?)?;
                            let v_next = ((params.beta_2 * v.as_tensor())?
                                + ((1. - params.beta_2) * grad.powf(2.)?)?)?;
                            let m_hat = (&m_next / (1. - (params.beta_1).powf(t)))?;
                            let vmax_next = vmax.maximum(&v_next)?;
                            let v_hat = (&vmax_next / (1. - params.beta_2.powf(t)))?;
                            let delta =
                                (m_hat * params.lr)?.div(&(v_hat.powf(0.5)? + params.eps)?)?;
                            theta.set(&theta.sub(&(delta))?)?;
                            m.set(&m_next)?;
                            v.set(&v_next)?;
                            vmax.set(&vmax_next)?;
                        }
                    }
                }
            }
        } else {
            for var in &self.0 {
                let theta = &var.theta;
                let m = &var.m;
                let v = &var.v;
                let vmax = &var.vmax;
                if let Some(grad) = grads.get(theta) {
                    let m_next =
                        ((params.beta_1 * m.as_tensor())? + ((1. - params.beta_1) * grad)?)?;
                    let v_next = ((params.beta_2 * v.as_tensor())?
                        + ((1. - params.beta_2) * grad.powf(2.)?)?)?;
                    let m_hat = (&m_next / (1. - (params.beta_1).powf(t)))?;
                    let vmax_next = vmax.maximum(&v_next)?;
                    let v_hat = (&vmax_next / (1. - params.beta_2.powf(t)))?;
                    let delta = (m_hat * params.lr)?.div(&(v_hat.powf(0.5)? + params.eps)?)?;
                    theta.set(&theta.sub(&(delta))?)?;
                    m.set(&m_next)?;
                    v.set(&v_next)?;
                    vmax.set(&vmax_next)?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
enum VarAdam {
    VecAdamBase(VecAdamBase),
    VecAdamAmsgrad(VecAdamAmsgrad),
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug)]
pub struct ParamsAdam {
    pub lr: f64,
    pub beta_1: f64,
    pub beta_2: f64,
    pub eps: f64,
    pub weight_decay: Option<Decay>,
    pub amsgrad: bool,
    // pub decoupled_weight_decay: bool,
}

impl Default for ParamsAdam {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta_1: 0.9,
            beta_2: 0.999,
            eps: 1e-8,
            weight_decay: None,
            amsgrad: false,
            // decoupled_weight_decay: false,
        }
    }
}

impl Optimizer for Adam {
    type Config = ParamsAdam;

    fn new(vars: Vec<Var>, params: ParamsAdam) -> Result<Self> {
        if params.amsgrad {
            let vars = VarAdam::VecAdamAmsgrad(VecAdamAmsgrad::new(vars)?);
            Ok(Self {
                vars,
                params,
                t: 1.,
            })
        } else {
            let vars = VarAdam::VecAdamBase(VecAdamBase::new(vars)?);
            Ok(Self {
                vars,
                params,
                t: 1.,
            })
        }
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        match &self.vars {
            VarAdam::VecAdamBase(vars) => vars.inner_step(&self.params, grads, self.t)?,
            VarAdam::VecAdamAmsgrad(vars) => vars.inner_step(&self.params, grads, self.t)?,
        }
        self.t += 1.;
        Ok(())
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }
}

impl Adam {
    #[must_use]
    pub fn into_inner(self) -> Vec<Var> {
        match self.vars {
            VarAdam::VecAdamBase(vars) => vars.into_inner(),
            VarAdam::VecAdamAmsgrad(vars) => vars.into_inner(),
        }
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
        let params = ParamsAdam {
            lr: 0.004,
            ..Default::default()
        };
        // Now use backprop to run a linear regression between samples and get the coefficients back.
        let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
        let b = Var::new(0f32, &Device::Cpu)?;
        let mut n_sgd = Adam::new(vec![w.clone(), b.clone()], params)?;
        assert_approx_eq!(0.004, n_sgd.learning_rate());
        n_sgd.set_learning_rate(0.002);
        assert_approx_eq!(0.002, n_sgd.learning_rate());
        Ok(())
    }

    #[test]
    fn into_inner_test() -> Result<()> {
        let params = ParamsAdam::default();
        let w = Var::new(&[[3f32, 1.]], &Device::Cpu)?;
        let b = Var::new(-2f32, &Device::Cpu)?;
        let n_sgd = Adam::new(vec![w.clone(), b.clone()], params)?;
        let inner = n_sgd.into_inner();
        assert_eq!(inner[0].as_tensor().to_vec2::<f32>()?, &[[3f32, 1.]]);
        assert_approx_eq!(inner[1].as_tensor().to_vec0::<f32>()?, -2_f32);
        let params = ParamsAdam {
            amsgrad: true,
            ..Default::default()
        };
        let w = Var::new(&[[3f32, 1.]], &Device::Cpu)?;
        let b = Var::new(-2f32, &Device::Cpu)?;
        let n_sgd = Adam::new(vec![w.clone(), b.clone()], params)?;
        let inner = n_sgd.into_inner();
        assert_eq!(inner[0].as_tensor().to_vec2::<f32>()?, &[[3f32, 1.]]);
        assert_approx_eq!(inner[1].as_tensor().to_vec0::<f32>()?, -2_f32);
        Ok(())
    }
}
