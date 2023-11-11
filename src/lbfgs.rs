use crate::{LossOptimizer, Model};
use candle_core::Result as CResult;
use candle_core::{backprop::GradStore, Device, Tensor, Var};
use candle_nn::optim::Optimizer;

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
    pub tolerance_grad: f64,
    pub tolerance_change: f64,
    pub history_size: usize,
    pub line_search: Option<LineSearch>,
}

impl Default for ParamsLBFGS {
    fn default() -> Self {
        Self {
            lr: 1.,
            max_iter: 20,
            max_eval: None,
            tolerance_grad: 1e-7,
            tolerance_change: 1e-9,
            history_size: 100,
            line_search: None,
        }
    }
}

#[derive(Debug)]
pub struct Lbfgs<M: Model> {
    vars: Vec<Var>,
    model: M,
    s_hist: Vec<Tensor>,
    y_hist: Vec<Tensor>,
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
            s_hist: Vec::with_capacity(hist_size),
            y_hist: Vec::with_capacity(hist_size),
            params,
        })
    }

    fn backward_step(&mut self, xs: &Tensor, ys: &Tensor) -> CResult<()> {
        let loss = self.model.loss(xs, ys)?;
        let grads = loss.backward()?;
        let flat_grads = flatten_grads(self.vars.clone(), grads)?;
        let max_force = flat_grads.abs()?.max(0)?.to_scalar::<f64>()?;
        if max_force < self.params.tolerance_grad {
            println!("Early exit: force convergence");
            return Ok(());
        }
        let mut evals = 1;

        todo!()
    }

    fn learning_rate(&self) -> f64 {
        todo!()
    }

    fn set_learning_rate(&mut self, lr: f64) {
        todo!()
    }

    fn into_inner(self) -> Vec<Var> {
        todo!()
    }
}

fn flatten_grads(vs: Vec<Var>, grads: GradStore) -> CResult<Tensor> {
    let mut flat_grads = Vec::new();
    for v in vs {
        if let Some(grad) = grads.get(&v) {
            flat_grads.push(grad.reshape(((), 1, 1, 1))?);
        } else {
            let n_elems = v.elem_count();
            flat_grads.push(candle_core::Tensor::zeros(
                (n_elems, 1, 1, 1),
                v.dtype(),
                v.device(),
            )?);
        }
    }
    candle_core::Tensor::cat(&flat_grads, 0)
}
