use std::collections::VecDeque;

use crate::{LossOptimizer, Model};
use candle_core::Result as CResult;
use candle_core::{Tensor, Var};
// use candle_nn::optim::Optimizer;

mod strong_wolfe;

#[derive(Debug)]
pub enum LineSearch {
    StrongWolfe,
}

/// LBFGS optimiser: see Nocedal
///
/// Described in <https://link.springer.com/article/10.1007/BF01589116>]
///
/// https://sagecal.sourceforge.net/pytorch/index.html
/// https://github.com/hjmshi/PyTorch-LBFGS/blob/master/functions/LBFGS.py

#[derive(Debug)]
pub struct ParamsLBFGS {
    pub lr: f64,
    pub max_iter: usize,
    pub max_eval: Option<usize>,
    pub history_size: usize,
    pub line_search: Option<LineSearch>,
}

impl Default for ParamsLBFGS {
    fn default() -> Self {
        Self {
            lr: 1.,
            max_iter: 20,
            max_eval: None,
            history_size: 100,
            line_search: None,
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
    // avg_acc: HashMap<TensorId, (Tensor, Tensor)>,
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

    fn backward_step(&mut self, xs: &Tensor, ys: &Tensor) -> CResult<()> {
        let loss = self.model.loss(xs, ys)?;
        println!("loss: {}", loss);
        // let grads = loss.backward()?;

        // let mut evals = 1;
        let grad = flat_grads(&self.vars, loss)?;
        println!("grad: {}", grad);
        println!(
            "max F: {}",
            grad.abs()?
                .max(0)?
                .to_dtype(candle_core::DType::F64)?
                .to_scalar::<f64>()?
        );
        if grad
            .abs()?
            .max(0)?
            .to_dtype(candle_core::DType::F64)?
            .to_scalar::<f64>()?
            < 1e-6
        {
            println!("grad is small enough");
            return Ok(());
        }

        let mut yk = None;

        if let Some(last) = &self.last_grad {
            yk = Some((&grad - last.as_tensor())?);
            last.set(&grad)?;
        } else {
            self.last_grad = Some(Var::from_tensor(&grad)?);
        }

        println!("grad: {}", grad);

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
        // self.s_hist.push_back((q.into_inner(), yk));

        let gamma = if let Some((s, y)) = self.s_hist.back() {
            println!("y: {}", y);
            println!("s: {}", s);
            let numr = (y * s)?
                .sum_all()?
                .to_dtype(candle_core::DType::F64)?
                .to_scalar::<f64>()?;
            let denom = &y
                .sqr()?
                .sum_all()?
                .to_dtype(candle_core::DType::F64)?
                .to_scalar::<f64>()?
                + 1e-10; // add a little to avoid divide by zero
            println!("numr: {}", numr);
            println!("denom: {}", denom);
            numr / denom
        } else {
            1. // self.learning_rate()
        };
        println!("gamma: {}", gamma);

        let mut rhos = VecDeque::with_capacity(hist_size);
        let mut alphas = VecDeque::with_capacity(hist_size);
        for (s, y) in self.s_hist.iter().rev() {
            // alt dot product? println!("test: {}", test);
            // let test = y
            //     .reshape((1, ()))?
            //     .matmul(&s.reshape(((), 1))?)?
            //     .to_dtype(candle_core::DType::F64)?
            //     .reshape(())?
            //     .to_scalar::<f64>()?
            //     .powf(-1.);
            let rho = ((y * s)?
                .sum_all()?
                .to_dtype(candle_core::DType::F64)?
                .to_scalar::<f64>()?
                + 1e-10)
                .powi(-1);
            println!("rho: {}", rho);

            let alpha = &rho
                * (s * q.as_tensor())?
                    .sum_all()?
                    .to_dtype(candle_core::DType::F64)?
                    .to_scalar::<f64>()?;

            q.set(&q.sub(&(y * alpha)?)?)?;
            // println!("alpha: {}", alpha);
            // println!("rho: {}", rho);
            // we are iterating in reverse and so want to insert at the front of the VecDeque
            alphas.push_front(alpha);
            rhos.push_front(rho);
        }
        println!("q after loop 1: {}", q);

        // z = q * gamma so use interior mutability of q to set it
        q.set(&(q.as_tensor() * gamma)?)?;
        println!("q before loop 2: {}", q);
        for (((s, y), alpha), rho) in self
            .s_hist
            .iter()
            .zip(alphas.into_iter())
            .zip(rhos.into_iter())
        {
            // println!("alpha: {}", alpha);
            // println!("rho: {}", rho);
            let beta = rho
                * (y * q.as_tensor())?
                    .sum_all()?
                    .to_dtype(candle_core::DType::F64)?
                    .to_scalar::<f64>()?;
            // println!("beta: {}", beta);
            q.set(&q.add(&(s * (alpha - beta))?)?)?;
            // println!("q: {}", q);
        }

        println!("q after loop 2: {}", q);

        let dd = (&grad * q.as_tensor())?.sum_all()?;
        println!("dd: {}", dd);
        if dd.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()? < 0. {
            println!("WARN: Maximising step");
            // q.set(&(q.as_tensor() * -1.)?)?; // flip the sign

            // if let Some(ref last) = self.last_grad {
            //     yk = ((-1. * &grad)? - last.as_tensor())?;
            //     last.set(&(-1. * &grad)?)?;
            // } else {
            //     self.last_grad = Some(Var::from_tensor(&(-1. * &grad)?)?);
            //     yk = (-1. * grad.copy()?)?
            // };

            // if let Some(ref last) = self.last_grad {
            //     yk = (&grad - last.as_tensor())?;
            //     last.set(&grad)?;
            // } else {
            //     self.last_grad = Some(Var::from_tensor(&grad)?);
            //     yk = grad.copy()?
            // };
        }
        // else {
        //     if let Some(ref last) = self.last_grad {
        //         yk = (&grad - last.as_tensor())?;
        //         last.set(&grad)?;
        //     } else {
        //         self.last_grad = Some(Var::from_tensor(&grad)?);
        //         yk = grad.copy()?
        //     };
        // }

        // println!("yk: {}", yk);
        let lr = if self.first {
            self.first = false;
            1_f64.min(
                1. / (&grad)
                    .abs()?
                    .sum_all()?
                    .to_dtype(candle_core::DType::F64)?
                    .to_scalar::<f64>()?,
            ) * self.params.lr
        } else {
            self.params.lr
        };
        println!("lr : {lr}");

        q.set(&(q.as_tensor() * -lr)?)?;

        println!("step: {}", q);

        if let Some(step) = &self.last_step {
            step.set(&q)?;
        } else {
            self.last_step = Some(Var::from_tensor(&q)?);
        }

        add_grad(&mut self.vars, &q.as_tensor())?;

        for v in &self.vars {
            println!("end of iter: {}", v);
        }

        Ok(())
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

fn flat_grads(vs: &Vec<Var>, loss: Tensor) -> CResult<Tensor> {
    let grads = loss.backward()?;
    let mut flat_grads = Vec::with_capacity(vs.len());
    for v in vs {
        if let Some(grad) = grads.get(&v) {
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

#[cfg(test)]
mod tests {
    // use candle_core::test_utils::{to_vec0_round, to_vec2_round};

    use crate::Model;
    use anyhow::Result;
    use assert_approx_eq::assert_approx_eq;
    use candle_core::{DType, Device};
    use candle_core::{Module, Result as CResult};
    use candle_nn::{VarBuilder, VarMap};

    use super::*;
    #[test]
    fn lr_test() -> Result<()> {
        let params = ParamsLBFGS {
            lr: 0.004,
            ..Default::default()
        };
        // Now use backprop to run a linear regression between samples and get the coefficients back.
        pub struct LinearModel {
            linear: candle_nn::Linear,
        }

        impl Model for LinearModel {
            fn loss(&self, xs: &Tensor, ys: &Tensor) -> CResult<Tensor> {
                let preds = self.forward(xs)?;
                let loss = candle_nn::loss::mse(&preds, ys)?;
                Ok(loss)
            }
        }

        impl LinearModel {
            fn new(vs: VarBuilder) -> CResult<Self> {
                let linear = candle_nn::linear(2, 1, vs.pp("ln1"))?;
                Ok(Self { linear })
            }

            fn forward(&self, xs: &Tensor) -> CResult<Tensor> {
                self.linear.forward(xs)
            }
        }

        // create a new variable store
        let varmap = VarMap::new();
        // create a new variable builder
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let model = LinearModel::new(vs)?;
        let mut lbfgs = Lbfgs::new(varmap.all_vars(), params, model)?;
        assert_approx_eq!(0.004, lbfgs.learning_rate());
        lbfgs.set_learning_rate(0.002);
        assert_approx_eq!(0.002, lbfgs.learning_rate());
        Ok(())
    }
}
