#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

// use candle_core::test_utils::{to_vec0_round, to_vec2_round};

use anyhow::Result;
use candle_core::{DType, Result as CResult};
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap};
use optimisers::lbfgs::{Lbfgs, ParamsLBFGS};
use optimisers::{LossOptimizer, Model};

/* The results of this test have been checked against the following PyTorch code.
    import torch
    from torch import optim

    w_gen = torch.tensor([[3., 1.]])
    b_gen = torch.tensor([-2.])

    sample_xs = torch.tensor([[2., 1.], [7., 4.], [-4., 12.], [5., 8.]])
    sample_ys = sample_xs.matmul(w_gen.t()) + b_gen

    m = torch.nn.Linear(2, 1)
    with torch.no_grad():
        m.weight.zero_()
        m.bias.zero_()
    optimiser = optim.Adamax(m.parameters(), lr=0.004)
    # optimiser.zero_grad()
    for _step in range(100):
        optimiser.zero_grad()
        ys = m(sample_xs)
        loss = ((ys - sample_ys)**2).sum()
        loss.backward()
        optimiser.step()
        # print("Optimizer state begin")
        # print(optimiser.state)
        # print("Optimizer state end")
    print(m.weight)
    print(m.bias)
*/
#[test]
fn lbfgs_test() -> Result<()> {
    // Generate some linear data, y = 3.x1 + x2 - 2.
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
    let sample_ys = gen.forward(&sample_xs)?;

    pub struct LinearModel {
        linear: candle_nn::Linear,
    }

    impl Model for LinearModel {
        fn new(vs: VarBuilder) -> CResult<Self> {
            let linear = candle_nn::linear(2, 1, vs.pp("ln1"))?;
            Ok(Self { linear })
        }

        fn forward(&self, xs: &Tensor) -> CResult<Tensor> {
            self.linear.forward(xs)
        }
        fn loss(&self, xs: &Tensor, ys: &Tensor) -> CResult<Tensor> {
            let preds = self.forward(xs)?;
            let loss = candle_nn::loss::mse(&preds, ys)?;
            Ok(loss)
        }
    }

    let params = ParamsLBFGS {
        lr: 0.004,
        ..Default::default()
    };

    // create a new variable store
    let varmap = VarMap::new();
    // create a new variable builder
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let model = LinearModel::new(vs)?;
    let mut lbfgs = Lbfgs::new(varmap.all_vars(), params, model)?;

    for _step in 0..500_000 {
        // println!("start step {}", _step);
        lbfgs.backward_step(&sample_xs, &sample_ys)?;
        // println!("end step {}", _step);
    }
    // panic!("stop");

    Ok(())
}
