use std::collections::VecDeque;
use std::ops::Add;

use crate::{LossOptimizer, Model};
use candle_core::Result as CResult;
use candle_core::{backprop::GradStore, Tensor, Var};
// use candle_nn::optim::Optimizer;

mod strong_wolfe;

#[derive(Debug)]
pub enum LineSearch {
    StrongWolfe,
}

/// LBFGS optimiser
///
/// Described in <https://link.springer.com/article/10.1007/BF01589116>]

pub trait ModelLoss: Sized {
    fn get_loss(&self, xs: &Tensor) -> CResult<Tensor>;
}

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
    hist: VecDeque<(Tensor, Tensor)>,
    last_grad: Option<Var>,
    params: ParamsLBFGS,
    // avg_acc: HashMap<TensorId, (Tensor, Tensor)>,
}

impl<M: Model> LossOptimizer<M> for Lbfgs<M> {
    type Config = ParamsLBFGS;

    fn new(vs: Vec<Var>, params: Self::Config, model: M) -> CResult<Self> {
        let hist_size = params.history_size;
        Ok(Lbfgs {
            vars: vs,
            model,
            hist: VecDeque::with_capacity(hist_size),
            last_grad: None,
            params,
        })
    }

    fn backward_step(&mut self, xs: &Tensor, ys: &Tensor) -> CResult<()> {
        let loss = self.model.loss(xs, ys)?;
        let grads = loss.backward()?;
        // let mut evals = 1;
        let mut q = flatten_grads(self.vars.clone(), grads)?;
        let yk = if let Some(ref last) = self.last_grad {
            last.set(&q)?;
            (&q - last.as_tensor())?
        } else {
            self.last_grad = Some(Var::from_tensor(&q)?);
            q.clone()
        };
        let hist_size = self.hist.len();
        let gamma = if let Some((s, y)) = self.hist.back() {
            let numr = (y * s)?.sum_all()?;
            let denom = y.sqr()?.sum_all()?;
            (numr / denom)?.to_scalar::<f64>()?
        } else {
            self.learning_rate()
        };
        let mut rhos = Vec::with_capacity(hist_size);
        let mut alphas = Vec::with_capacity(hist_size);
        for (s, y) in self.hist.iter() {
            let rho = (y * s)?.sum_all()?.powf(-1.)?;
            let alpha = (&rho * (s * &q)?.sum_all()?)?;
            q = q.sub(&(y * &alpha)?)?;
            alphas.push(alpha);
            rhos.push(rho);
        }
        q = (q * gamma)?;
        for (((s, y), alpha), rho) in self.hist.iter().zip(alphas).zip(rhos) {
            let beta = rho * (y * &q)?.sum_all()?;
            q = q.add(&(s * (alpha - beta)?)?)?;
        }
        add_grad(&mut self.vars, &q)?;

        if hist_size == self.params.history_size {
            self.hist.pop_front();
        }
        self.hist.push_back((q, yk));

        todo!()
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    fn into_inner(self) -> Vec<Var> {
        todo!()
    }
}

fn flatten_grads(vs: Vec<Var>, grads: GradStore) -> CResult<Tensor> {
    let mut flat_grads = Vec::new();
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
        let n_elems = (&var).elem_count();
        let tensor = flat_tensor
            .narrow(0, offset, n_elems)?
            .reshape((&var).shape())?;
        var.set(&var.sub(&tensor)?)?;
        offset += n_elems;
    }
    Ok(())
}

// fn unflatten(vs: Vec<Var>, flat_tensor: Tensor) -> CResult<Vec<Tensor>> {
//     let mut tensors = Vec::new();
//     let mut offset = 0;
//     for var in vs {
//         let n_elems = var.elem_count();
//         let tensor = flat_tensor
//             .narrow(0, offset, n_elems)?
//             .reshape(var.shape())?;
//         tensors.push(tensor);
//         offset += n_elems;
//     }
//     Ok(tensors)
// }
