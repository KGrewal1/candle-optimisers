#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

// use candle_core::test_utils::{to_vec0_round, to_vec2_round};

use anyhow::Result;
use candle_core::test_utils::to_vec2_round;
use candle_core::{DType, Result as CResult};
use candle_core::{Device, Tensor};
// use candle_nn::{Linear, Module, VarBuilder, VarMap};
use optimisers::lbfgs::{Lbfgs, LineSearch, ParamsLBFGS};
use optimisers::{LossOptimizer, Model};

/*
These tests all use the 2D Rosenbrock function as a test function for the optimisers. This has minimum 0 at (1, 1)
*/

#[test]
fn lbfgs_test() -> Result<()> {
    #[derive(Debug, Clone)]
    pub struct RosenbrockModel {
        x_pos: candle_core::Var,
        y_pos: candle_core::Var,
    }

    impl Model for RosenbrockModel {
        fn loss(&self) -> CResult<Tensor> {
            //, xs: &Tensor, ys: &Tensor
            self.forward()?.squeeze(1)?.squeeze(0)
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
    let mut loss = model.loss()?;

    for _step in 0..500 {
        // println!("\nstart step {}", step);
        // for v in model.vars() {
        //     println!("{}", v);
        // }
        let res = lbfgs.backward_step(&loss)?; //&sample_xs, &sample_ys
                                               // println!("end step {}", _step);
        match res {
            optimisers::ModelOutcome::Converged(_, _) => break,
            optimisers::ModelOutcome::Stepped(new_loss, _) => loss = new_loss,
            // _ => panic!("unexpected outcome"),
        }
    }

    for v in model.vars() {
        // println!("{}", v);
        assert_eq!(to_vec2_round(&v.to_dtype(DType::F32)?, 4)?, &[[1.0000]]);
    }

    // println!("{:?}", lbfgs);
    // panic!("deliberate panic");

    Ok(())
}

#[test]
fn lbfgs_test_strong_wolfe() -> Result<()> {
    #[derive(Debug, Clone)]
    pub struct RosenbrockModel {
        x_pos: candle_core::Var,
        y_pos: candle_core::Var,
    }

    impl Model for RosenbrockModel {
        fn loss(&self) -> CResult<Tensor> {
            //, xs: &Tensor, ys: &Tensor
            self.forward()?.squeeze(1)?.squeeze(0)
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
        line_search: Some(LineSearch::StrongWolfe),
        ..Default::default()
    };

    let model = RosenbrockModel::new()?;

    let mut lbfgs = Lbfgs::new(model.vars(), params, model.clone())?;
    let mut loss = model.loss()?;

    for _step in 0..500 {
        // println!("\nstart step {}", step);
        // for v in model.vars() {
        //     println!("{}", v);
        // }
        let res = lbfgs.backward_step(&loss)?; //&sample_xs, &sample_ys
                                               // println!("end step {}", _step);
        match res {
            optimisers::ModelOutcome::Converged(_, _) => break,
            optimisers::ModelOutcome::Stepped(new_loss, _) => loss = new_loss,
            // _ => panic!("unexpected outcome"),
        }
    }

    for v in model.vars() {
        // println!("{}", v);
        assert_eq!(to_vec2_round(&v.to_dtype(DType::F32)?, 4)?, &[[1.0000]]);
    }

    // println!("{:?}", lbfgs);
    // panic!("deliberate panic");

    Ok(())
}
