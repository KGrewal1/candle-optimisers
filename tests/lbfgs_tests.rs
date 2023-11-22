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
    let w_gen = Tensor::new(&[[3f64, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f64, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f64, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
    let sample_ys = gen.forward(&sample_xs)?;

    #[derive(Debug, Clone)]
    pub struct RosenbrockModel {
        x_pos: candle_core::Var,
        y_pos: candle_core::Var,
    }

    impl Model for RosenbrockModel {
        fn loss(&self) -> CResult<Tensor> {
            //, xs: &Tensor, ys: &Tensor
            self.forward()
        }
    }

    impl RosenbrockModel {
        fn new() -> CResult<Self> {
            let x_pos = candle_core::Var::from_tensor(
                &(10. * Tensor::ones((1, 1), DType::F64, &Device::Cpu)?)?,
            )?;
            let y_pos = candle_core::Var::from_tensor(
                &(10. * Tensor::ones((1, 1), DType::F64, &Device::Cpu)?)?,
            )?;
            Ok(Self { x_pos, y_pos })
        }
        fn vars(&self) -> Vec<candle_core::Var> {
            vec![self.x_pos.clone(), self.y_pos.clone()]
        }

        fn forward(&self) -> CResult<Tensor> {
            //, xs: &Tensor
            (1. - self.x_pos.as_tensor())?.powf(2.)?
                + 100. * (self.y_pos.as_tensor() - self.x_pos.as_tensor().powf(2.)?)?.powf(2.)?
        }
    }

    let params = ParamsLBFGS {
        lr: 1.,
        ..Default::default()
    };

    let model = RosenbrockModel::new()?;

    let mut lbfgs = Lbfgs::new(model.vars(), params, model.clone())?;

    for step in 0..500 {
        println!("\nstart step {}", step);
        for v in model.vars() {
            println!("{}", v);
        }
        lbfgs.backward_step()?; //&sample_xs, &sample_ys
                                // println!("end step {}", _step);
    }
    for v in model.vars() {
        println!("{}", v);
    }
    // println!("{:?}", lbfgs);
    panic!("deliberate panic");

    Ok(())
}
