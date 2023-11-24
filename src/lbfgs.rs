use std::collections::VecDeque;

use crate::{LossOptimizer, Model, ModelOutcome};
use candle_core::Result as CResult;
use candle_core::{Tensor, Var};
// use candle_nn::optim::Optimizer;

mod strong_wolfe;

#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum LineSearch {
    /// strong wolfe line search: c1, c2, tolerance
    /// suggested vals for c1 and c2: 1e-4, 0.9
    StrongWolfe(f64, f64, f64),
}

#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum GradConv {
    /// convergence based on max abs component of gradient
    MinForce(f64),
    /// convergence based on mean force
    RMSForce(f64),
}

#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum StepConv {
    /// convergence based on max abs component of step
    MinStep(f64),
    /// convergence based on root mean size of step
    RMSStep(f64),
}

/// LBFGS optimiser: see Nocedal
///
/// Described in <https://link.springer.com/article/10.1007/BF01589116>]
///
/// <https://sagecal.sourceforge.net/pytorch/index.html>
/// <https://github.com/hjmshi/PyTorch-LBFGS/blob/master/functions/LBFGS.py>

#[derive(Clone, Copy, Debug)]
pub struct ParamsLBFGS {
    pub lr: f64,
    // pub max_iter: usize,
    // pub max_eval: Option<usize>,
    pub history_size: usize,
    pub line_search: Option<LineSearch>,
    pub grad_conv: GradConv,
    pub step_conv: StepConv,
}

impl Default for ParamsLBFGS {
    fn default() -> Self {
        Self {
            lr: 1.,
            // max_iter: 20,
            // max_eval: None,
            history_size: 100,
            line_search: None,
            grad_conv: GradConv::MinForce(1e-7),
            step_conv: StepConv::MinStep(1e-9),
        }
    }
}

#[derive(Debug)]
pub struct Lbfgs<M: Model> {
    vars: Vec<Var>,
    model: M,
    s_hist: VecDeque<(Tensor, Tensor)>,
    last_grad: Option<Var>,
    last_step: Option<Var>,
    params: ParamsLBFGS,
    first: bool,
}

impl<M: Model> LossOptimizer<M> for Lbfgs<M> {
    type Config = ParamsLBFGS;

    fn new(vs: Vec<Var>, params: Self::Config, model: M) -> CResult<Self> {
        let hist_size = params.history_size;
        Ok(Lbfgs {
            vars: vs,
            model,
            s_hist: VecDeque::with_capacity(hist_size),
            last_step: None,
            last_grad: None,
            params,
            first: true,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn backward_step(&mut self, loss: &Tensor) -> CResult<ModelOutcome> {
        let mut evals = 1;
        let grad = flat_grads(&self.vars, loss)?;

        match self.params.grad_conv {
            GradConv::MinForce(tol) => {
                if grad
                    .abs()?
                    .max(0)?
                    .to_dtype(candle_core::DType::F64)?
                    .to_scalar::<f64>()?
                    < tol
                {
                    return Ok(ModelOutcome::Converged(loss.clone(), evals));
                }
            }
            GradConv::RMSForce(tol) => {
                if grad
                    .sqr()?
                    .mean_all()?
                    .to_dtype(candle_core::DType::F64)?
                    .to_scalar::<f64>()?
                    .sqrt()
                    < tol
                {
                    return Ok(ModelOutcome::Converged(loss.clone(), evals));
                }
            }
        }

        let mut yk = None;

        if let Some(last) = &self.last_grad {
            yk = Some((&grad - last.as_tensor())?);
            last.set(&grad)?;
        } else {
            self.last_grad = Some(Var::from_tensor(&grad)?);
        }

        let q = Var::from_tensor(&grad)?;

        let hist_size = self.s_hist.len();

        if hist_size == self.params.history_size {
            self.s_hist.pop_front();
        }
        if let Some(yk) = yk {
            if let Some(step) = &self.last_step {
                self.s_hist.push_back((step.as_tensor().clone(), yk));
            }
        }

        let gamma = if let Some((s, y)) = self.s_hist.back() {
            let numr = y
                .unsqueeze(0)?
                .matmul(&(s.unsqueeze(1)?))?
                .to_dtype(candle_core::DType::F64)?
                .sum_all()?
                .to_scalar::<f64>()?;

            let denom = y
                .unsqueeze(0)?
                .matmul(&(y.unsqueeze(1)?))?
                .to_dtype(candle_core::DType::F64)?
                .sum_all()?
                .to_scalar::<f64>()?
                + 1e-10;

            numr / denom
        } else {
            1.
        };

        let mut rhos = VecDeque::with_capacity(hist_size);
        let mut alphas = VecDeque::with_capacity(hist_size);
        for (s, y) in self.s_hist.iter().rev() {
            let rho = (y
                .unsqueeze(0)?
                .matmul(&(s.unsqueeze(1)?))?
                .to_dtype(candle_core::DType::F64)?
                .sum_all()?
                .to_scalar::<f64>()?
                + 1e-10)
                .powi(-1);

            let alpha = rho
                * s.unsqueeze(0)?
                    .matmul(&(q.unsqueeze(1)?))?
                    .to_dtype(candle_core::DType::F64)?
                    .sum_all()?
                    .to_scalar::<f64>()?;

            q.set(&q.sub(&(y * alpha)?)?)?;
            // we are iterating in reverse and so want to insert at the front of the VecDeque
            alphas.push_front(alpha);
            rhos.push_front(rho);
        }

        // z = q * gamma so use interior mutability of q to set it
        q.set(&(q.as_tensor() * gamma)?)?;
        for (((s, y), alpha), rho) in self
            .s_hist
            .iter()
            .zip(alphas.into_iter())
            .zip(rhos.into_iter())
        {
            let beta = rho
                * y.unsqueeze(0)?
                    .matmul(&(q.unsqueeze(1)?))?
                    .to_dtype(candle_core::DType::F64)?
                    .sum_all()?
                    .to_scalar::<f64>()?;

            q.set(&q.add(&(s * (alpha - beta))?)?)?;
        }

        // let dd = (&grad * q.as_tensor())?.sum_all()?;
        let dd = grad
            .unsqueeze(0)?
            .matmul(&(q.unsqueeze(1)?))?
            .to_dtype(candle_core::DType::F64)?
            .sum_all()?
            .to_scalar::<f64>()?;

        let mut lr = if self.first {
            self.first = false;
            -(1_f64.min(
                1. / grad
                    .abs()?
                    .sum_all()?
                    .to_dtype(candle_core::DType::F64)?
                    .to_scalar::<f64>()?,
            )) * self.params.lr
        } else {
            -self.params.lr
        };

        if let Some(ls) = &self.params.line_search {
            match ls {
                LineSearch::StrongWolfe(c1, c2, tol) => {
                    let (_loss, _grad, t, steps) = self.strong_wolfe(
                        lr,
                        &q,
                        loss.to_dtype(candle_core::DType::F64)?.to_scalar()?,
                        &grad,
                        dd,
                        *c1,
                        *c2,
                        *tol,
                        25,
                    )?;
                    evals += steps;
                    lr = t;
                }
            }
        }

        q.set(&(q.as_tensor() * lr)?)?;

        if let Some(step) = &self.last_step {
            step.set(&q)?;
        } else {
            self.last_step = Some(Var::from_tensor(&q)?);
        }

        match self.params.step_conv {
            StepConv::MinStep(tol) => {
                if q.abs()?
                    .max(0)?
                    .to_dtype(candle_core::DType::F64)?
                    .to_scalar::<f64>()?
                    < tol
                {
                    add_grad(&mut self.vars, q.as_tensor())?;

                    let next_loss = self.model.loss()?;
                    evals += 1;
                    Ok(ModelOutcome::Converged(next_loss, evals))
                } else {
                    add_grad(&mut self.vars, q.as_tensor())?;

                    let next_loss = self.model.loss()?;
                    evals += 1;
                    Ok(ModelOutcome::Stepped(next_loss, evals))
                }
            }
            StepConv::RMSStep(tol) => {
                if q.sqr()?
                    .mean_all()?
                    .to_dtype(candle_core::DType::F64)?
                    .to_scalar::<f64>()?
                    .sqrt()
                    < tol
                {
                    add_grad(&mut self.vars, q.as_tensor())?;

                    let next_loss = self.model.loss()?;
                    evals += 1;
                    Ok(ModelOutcome::Converged(next_loss, evals))
                } else {
                    add_grad(&mut self.vars, q.as_tensor())?;

                    let next_loss = self.model.loss()?;
                    evals += 1;
                    Ok(ModelOutcome::Stepped(next_loss, evals))
                }
            }
        }
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    #[must_use]
    fn into_inner(self) -> Vec<Var> {
        self.vars
    }
}

fn flat_grads(vs: &Vec<Var>, loss: &Tensor) -> CResult<Tensor> {
    let grads = loss.backward()?;
    let mut flat_grads = Vec::with_capacity(vs.len());
    for v in vs {
        if let Some(grad) = grads.get(v) {
            flat_grads.push(grad.flatten_all()?);
        } else {
            let n_elems = v.elem_count();
            flat_grads.push(candle_core::Tensor::zeros(n_elems, v.dtype(), v.device())?);
        }
    }
    candle_core::Tensor::cat(&flat_grads, 0)
}

fn add_grad(vs: &mut Vec<Var>, flat_tensor: &Tensor) -> CResult<()> {
    let mut offset = 0;
    for var in vs {
        let n_elems = var.elem_count();
        let tensor = flat_tensor
            .narrow(0, offset, n_elems)?
            .reshape(var.shape())?;
        var.set(&var.add(&tensor)?)?;
        offset += n_elems;
    }
    Ok(())
}

fn set_vs(vs: &mut [Var], vals: &Vec<Tensor>) -> CResult<()> {
    for (var, t) in vs.iter().zip(vals) {
        var.set(t)?;
    }
    Ok(())
}

impl<M: Model> Lbfgs<M> {
    fn directional_evaluate(&mut self, mag: f64, direction: &Tensor) -> CResult<(f64, Tensor)> {
        // need to cache the original result
        // Otherwise leads to drift over line search evals
        let original = self
            .vars
            .iter()
            .map(|v| v.as_tensor().copy())
            .collect::<CResult<Vec<Tensor>>>()?;

        add_grad(&mut self.vars, &(mag * direction)?)?;
        let loss = self.model.loss()?;
        let grad = flat_grads(&self.vars, &loss)?;
        set_vs(&mut self.vars, &original)?;
        // add_grad(&mut self.vars, &(-mag * direction)?)?;
        Ok((
            loss.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?,
            grad,
        ))
    }
}

#[cfg(test)]
mod tests {
    // use candle_core::test_utils::{to_vec0_round, to_vec2_round};

    use crate::Model;
    use anyhow::Result;
    use assert_approx_eq::assert_approx_eq;
    use candle_core::Device;
    use candle_core::{Module, Result as CResult};
    pub struct LinearModel {
        linear: candle_nn::Linear,
        xs: Tensor,
        ys: Tensor,
    }

    impl Model for LinearModel {
        fn loss(&self) -> CResult<Tensor> {
            let preds = self.forward(&self.xs)?;
            let loss = candle_nn::loss::mse(&preds, &self.ys)?;
            Ok(loss)
        }
    }

    impl LinearModel {
        fn new() -> CResult<(Self, Vec<Var>)> {
            let weight = Var::from_tensor(&Tensor::new(&[3f64, 1.], &Device::Cpu)?)?;
            let bias = Var::from_tensor(&Tensor::new(-2f64, &Device::Cpu)?)?;

            let linear =
                candle_nn::Linear::new(weight.as_tensor().clone(), Some(bias.as_tensor().clone()));

            Ok((
                Self {
                    linear,
                    xs: Tensor::new(&[[2f64, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?,
                    ys: Tensor::new(&[[7f64], [26.], [0.], [27.]], &Device::Cpu)?,
                },
                vec![weight, bias],
            ))
        }

        fn forward(&self, xs: &Tensor) -> CResult<Tensor> {
            self.linear.forward(xs)
        }
    }

    use super::*;
    #[test]
    fn lr_test() -> Result<()> {
        let params = ParamsLBFGS {
            lr: 0.004,
            ..Default::default()
        };
        let (model, vars) = LinearModel::new()?;
        let mut lbfgs = Lbfgs::new(vars, params, model)?;
        assert_approx_eq!(0.004, lbfgs.learning_rate());
        lbfgs.set_learning_rate(0.002);
        assert_approx_eq!(0.002, lbfgs.learning_rate());
        Ok(())
    }

    #[test]
    fn into_inner_test() -> Result<()> {
        let params = ParamsLBFGS {
            lr: 0.004,
            ..Default::default()
        };
        // Now use backprop to run a linear regression between samples and get the coefficients back.

        let (model, vars) = LinearModel::new()?;
        let slice: Vec<&Var> = vars.iter().collect();
        let lbfgs = Lbfgs::from_slice(&slice, params, model)?;
        let inner = lbfgs.into_inner();

        assert_eq!(inner[0].as_tensor().to_vec1::<f64>()?, &[3f64, 1.]);
        println!("checked weights");
        assert_approx_eq!(inner[1].as_tensor().to_vec0::<f64>()?, -2_f64);
        Ok(())
    }
}
