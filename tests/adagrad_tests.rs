#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle_core::test_utils::{to_vec0_round, to_vec2_round};

use anyhow::Result;
use candle_core::{Device, Tensor, Var};
use candle_nn::{Linear, Module, Optimizer};
use optimisers::adagrad::{Adagrad, ParamsAdaGrad};

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
    optimiser = optim.Adagrad(m.parameters(), lr=0.004, weight_decay=0.00)
    # optimiser.zero_grad()
    for _step in range(1000):
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
fn adagrad_test() -> Result<()> {
    // Generate some linear data, y = 3.x1 + x2 - 2.
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
    let sample_ys = gen.forward(&sample_xs)?;

    let params = ParamsAdaGrad {
        lr: 0.004,
        lr_decay: 0.0,
        weight_decay: 0.0,
        initial_acc: 0.0,
        eps: 1e-10,
    };
    // Now use backprop to run a linear regression between samples and get the coefficients back.
    let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
    let b = Var::new(0f32, &Device::Cpu)?;
    let mut n_sgd = Adagrad::new(vec![w.clone(), b.clone()], params)?;
    let lin = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));
    for _step in 0..1000 {
        let ys = lin.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        n_sgd.backward_step(&loss)?;
    }
    assert_eq!(to_vec2_round(&w, 4)?, &[[0.2424, 0.2341]]);
    assert_eq!(to_vec0_round(&b, 4)?, 0.2379);
    Ok(())
}

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
    optimiser = optim.Adagrad(m.parameters(), lr=0.004, lr_decay=0.2)
    # optimiser.zero_grad()
    for _step in range(1000):
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
fn adagrad_lr_decay_test() -> Result<()> {
    // Generate some linear data, y = 3.x1 + x2 - 2.
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
    let sample_ys = gen.forward(&sample_xs)?;

    let params = ParamsAdaGrad {
        lr: 0.004,
        lr_decay: 0.2,
        weight_decay: 0.0,
        initial_acc: 0.0,
        eps: 1e-10,
    };
    // Now use backprop to run a linear regression between samples and get the coefficients back.
    let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
    let b = Var::new(0f32, &Device::Cpu)?;
    let mut n_sgd = Adagrad::new(vec![w.clone(), b.clone()], params)?;
    let lin = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));
    for _step in 0..1000 {
        let ys = lin.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        n_sgd.backward_step(&loss)?;
    }
    assert_eq!(to_vec2_round(&w, 4)?, &[[0.0231, 0.0230]]);
    assert_eq!(to_vec0_round(&b, 4)?, 0.0230);
    Ok(())
}

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
    optimiser = optim.Adagrad(m.parameters(), lr=0.004, weight_decay=0.2)
    # optimiser.zero_grad()
    for _step in range(1000):
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
fn adagrad_weight_decay_test() -> Result<()> {
    // Generate some linear data, y = 3.x1 + x2 - 2.
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
    let sample_ys = gen.forward(&sample_xs)?;

    let params = ParamsAdaGrad {
        lr: 0.004,
        lr_decay: 0.0,
        weight_decay: 0.2,
        initial_acc: 0.0,
        eps: 1e-10,
    };
    // Now use backprop to run a linear regression between samples and get the coefficients back.
    let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
    let b = Var::new(0f32, &Device::Cpu)?;
    let mut n_sgd = Adagrad::new(vec![w.clone(), b.clone()], params)?;
    let lin = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));
    for _step in 0..1000 {
        let ys = lin.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        n_sgd.backward_step(&loss)?;
    }
    assert_eq!(to_vec2_round(&w, 4)?, &[[0.2424, 0.2341]]);
    assert_eq!(to_vec0_round(&b, 4)?, 0.2378);
    Ok(())
}
