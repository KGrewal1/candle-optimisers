use crate::Model;
use candle_core::Result as CResult;
use candle_core::{Tensor, Var};

use super::{add_grad, flat_grads, set_vs, Lbfgs};

/// ported from pytorch torch/optim/lbfgs.py ported from <https://github.com/torch/optim/blob/master/polyinterp.lua>
fn cubic_interpolate(
    // position 1
    x1: f64,
    // f(x1)
    f1: f64,
    // f'(x1)
    g1: f64,
    // position 2
    x2: f64,
    // f(x2)
    f2: f64,
    // f'(x2)
    g2: f64,
    bounds: Option<(f64, f64)>,
) -> f64 {
    let (xmin_bound, xmax_bound) = if let Some(bound) = bounds {
        bound
    } else if x1 < x2 {
        (x1, x2)
    } else {
        (x2, x1)
    };
    let d1 = g1 + g2 - 3. * (f1 - f2) / (x1 - x2);
    let d2_square = d1.powi(2) - g1 * g2;
    if d2_square >= 0. {
        let d2 = d2_square.sqrt();
        let min_pos = if x1 <= x2 {
            x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2. * d2))
        } else {
            x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2. * d2))
        };
        (min_pos.max(xmin_bound)).min(xmax_bound)
    } else {
        (xmin_bound + xmax_bound) / 2.
    }
}

impl<M: Model> Lbfgs<M> {
    /// Strong Wolfe line search
    ///
    /// # Arguments
    ///
    /// step size
    ///
    /// direction
    ///
    /// initial loss
    ///
    /// initial grad
    ///
    /// initial directional grad
    ///
    /// c1 coefficient for wolfe condition
    ///
    /// c2 coefficient for wolfe condition
    ///
    /// minimum allowed progress
    ///
    /// maximum number of iterations
    ///
    /// # Returns
    ///
    /// (`f_new`, `g_new`, t, `ls_func_evals`)
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(super) fn strong_wolfe(
        &mut self,
        mut step_size: f64,    // step size
        direction: &Tensor,    // direction
        loss: &Tensor,         // initial loss
        grad: &Tensor,         // initial grad
        directional_grad: f64, // initial directional grad
        c1: f64,               // c1 coefficient for wolfe condition
        c2: f64,               // c2 coefficient for wolfe condition
        tolerance_change: f64, // minimum allowed progress
        max_ls: usize,         // maximum number of iterations
    ) -> CResult<(Tensor, Tensor, f64, usize)> {
        // ported from https://github.com/torch/optim/blob/master/lswolfe.lua

        let d_norm = &direction
            .abs()?
            .max(0)?
            .to_dtype(candle_core::DType::F64)?
            .to_scalar::<f64>()?;

        // evaluate objective and gradient using initial step
        let (f_new, g_new, mut l2_new) = self.directional_evaluate(step_size, direction)?;
        let g_new = Var::from_tensor(&g_new)?;
        let mut f_new = f_new
            .to_dtype(candle_core::DType::F64)?
            .to_scalar::<f64>()?;
        let mut ls_func_evals = 1;
        let mut gtd_new = g_new
            .unsqueeze(0)?
            .matmul(&(direction.unsqueeze(1)?))?
            .to_dtype(candle_core::DType::F64)?
            .squeeze(1)?
            .squeeze(0)?
            .to_scalar::<f64>()?;

        // bracket an interval containing a point satisfying the Wolfe criteria
        let g_prev = Var::from_tensor(grad)?;
        let dtype = loss.dtype();
        let shape = loss.shape();
        let dev = loss.device();
        let mut f_prev = loss.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?;
        let l2_init = self.l2_reg()?;
        let mut l2_prev = l2_init;
        let (mut t_prev, mut gtd_prev) = (0., directional_grad);
        let mut done = false;
        let mut ls_iter = 0;

        let mut bracket_gtd;
        let mut bracket_l2;
        let mut bracket_f;
        let (mut bracket, bracket_g) = loop {
            // check conditions
            if f_new + l2_new >= f_prev + l2_prev {
                bracket_gtd = [gtd_prev, gtd_new];
                bracket_l2 = [l2_prev, l2_new];
                bracket_f = [f_prev, f_new];
                break (
                    [t_prev, step_size],
                    [g_prev, Var::from_tensor(g_new.as_tensor())?],
                );
            }

            if gtd_new.abs() <= -c2 * directional_grad {
                done = true;
                bracket_gtd = [gtd_prev, gtd_new];
                bracket_l2 = [l2_prev, l2_new];
                bracket_f = [f_new, f_new];
                break (
                    [step_size, step_size],
                    [
                        Var::from_tensor(g_new.as_tensor())?,
                        Var::from_tensor(g_new.as_tensor())?,
                    ],
                );
            }

            if gtd_new >= 0. {
                bracket_gtd = [gtd_prev, gtd_new];
                bracket_l2 = [l2_prev, l2_new];
                bracket_f = [f_prev, f_new];
                break (
                    [t_prev, step_size],
                    [g_prev, Var::from_tensor(g_new.as_tensor())?],
                );
            }

            // interpolate
            let min_step = step_size + 0.01 * (step_size - t_prev);
            let max_step = step_size * 10.;
            let tmp = step_size;
            step_size = cubic_interpolate(
                t_prev,
                f_prev + l2_prev,
                gtd_prev,
                step_size,
                f_new + l2_new,
                gtd_new,
                Some((min_step, max_step)),
            );

            // next step
            t_prev = tmp;
            f_prev = f_new;
            g_prev.set(g_new.as_tensor())?;
            l2_prev = l2_new;
            gtd_prev = gtd_new;
            // assign to temp vars:
            let (next_f, next_g, next_l2) = self.directional_evaluate(step_size, direction)?;

            // overwrite
            f_new = next_f
                .to_dtype(candle_core::DType::F64)?
                .to_scalar::<f64>()?;
            g_new.set(&next_g)?;
            l2_new = next_l2;

            ls_func_evals += 1;

            gtd_new = g_new
                .unsqueeze(0)?
                .matmul(&(direction.unsqueeze(1)?))?
                .to_dtype(candle_core::DType::F64)?
                .squeeze(1)?
                .squeeze(0)?
                .to_scalar::<f64>()?;
            ls_iter += 1;

            // reached max number of iterations?
            if ls_iter == max_ls {
                bracket_gtd = [gtd_prev, gtd_new];
                bracket_l2 = [l2_prev, l2_new];
                bracket_f = [
                    loss.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?,
                    f_new,
                ];
                break (
                    [0., step_size],
                    [
                        Var::from_tensor(grad)?,
                        Var::from_tensor(g_new.as_tensor())?,
                    ],
                );
            }
        };

        // zoom phase: we now have a point satisfying the criteria, or
        // a bracket around it. We refine the bracket until we find the
        // exact point satisfying the criteria
        let mut insuf_progress = false;
        // find high and low points in bracket
        let (mut low_pos, mut high_pos) =
            if bracket_f[0] + bracket_l2[0] <= bracket_f[1] + bracket_l2[1] {
                (0, 1)
            } else {
                (1, 0)
            };
        while !done && ls_iter < max_ls {
            // line-search bracket is so small
            if (bracket[1] - bracket[0]).abs() * d_norm < tolerance_change {
                break;
            }

            // compute new trial value
            step_size = cubic_interpolate(
                bracket[0],
                bracket_f[0] + bracket_l2[0],
                bracket_gtd[0],
                bracket[1],
                bracket_f[1] + bracket_l2[1],
                bracket_gtd[1],
                None,
            );

            // test that we are making sufficient progress:
            // in case `t` is so close to boundary, we mark that we are making
            // insufficient progress, and if
            //   + we have made insufficient progress in the last step, or
            //   + `t` is at one of the boundary,
            // we will move `t` to a position which is `0.1 * len(bracket)`
            // away from the nearest boundary point.
            let max_bracket = bracket[0].max(bracket[1]);
            let min_bracket = bracket[0].min(bracket[1]);
            let eps = 0.1 * (max_bracket - min_bracket);
            if (max_bracket - step_size).min(step_size - min_bracket) < eps {
                // interpolation close to boundary
                if insuf_progress || step_size >= max_bracket || step_size <= min_bracket {
                    // evaluate at 0.1 away from boundary
                    if (step_size - max_bracket).abs() < (step_size - min_bracket).abs() {
                        step_size = max_bracket - eps;
                    } else {
                        step_size = min_bracket + eps;
                    }
                    insuf_progress = false;
                } else {
                    insuf_progress = true;
                }
            } else {
                insuf_progress = false;
            }

            // Evaluate new point
            // assign to temp vars:
            let (next_f, next_g, next_l2) = self.directional_evaluate(step_size, direction)?;
            // overwrite
            f_new = next_f
                .to_dtype(candle_core::DType::F64)?
                .to_scalar::<f64>()?;
            g_new.set(&next_g)?;
            l2_new = next_l2;
            ls_func_evals += 1;

            gtd_new = g_new
                .unsqueeze(0)?
                .matmul(&(direction.unsqueeze(1)?))?
                .to_dtype(candle_core::DType::F64)?
                .squeeze(1)?
                .squeeze(0)?
                .to_scalar::<f64>()?;
            ls_iter += 1;

            if f_new + l2_new
                > (loss.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?
                    + l2_init
                    + c1 * step_size * directional_grad)
                || f_new + l2_new >= bracket_f[low_pos] + bracket_l2[low_pos]
            {
                // Armijo condition not satisfied or not lower than lowest point
                bracket[high_pos] = step_size;
                bracket_f[high_pos] = f_new;
                let _ = bracket_g[high_pos].set(g_new.as_tensor());
                bracket_l2[high_pos] = l2_new;
                bracket_gtd[high_pos] = gtd_new;

                (low_pos, high_pos) =
                    if bracket_f[0] + bracket_l2[0] <= bracket_f[1] + bracket_l2[1] {
                        (0, 1)
                    } else {
                        (1, 0)
                    };
            } else {
                if gtd_new.abs() <= -c2 * directional_grad {
                    // Wolfe conditions satisfied
                    done = true;
                } else if gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0. {
                    // old low becomes new high
                    bracket[high_pos] = bracket[low_pos];
                    bracket_f[high_pos] = bracket_f[low_pos];
                    bracket_g[high_pos].set(bracket_g[low_pos].as_tensor())?;
                    bracket_gtd[high_pos] = bracket_gtd[low_pos];
                    bracket_l2[high_pos] = bracket_l2[low_pos];
                }

                // new point becomes new low
                bracket[low_pos] = step_size;
                bracket_f[low_pos] = f_new;
                bracket_g[low_pos].set(g_new.as_tensor())?;
                bracket_gtd[low_pos] = gtd_new;
                bracket_l2[low_pos] = l2_new;
            }
        }

        // return new value, new grad, line-search value, nb of function evals
        step_size = bracket[low_pos];
        let [g0, g1] = bracket_g;
        let [f0, f1] = bracket_f;
        if low_pos == 1 {
            // if b is the lower value set a to b, else a should be returned
            Ok((
                Tensor::from_slice(&[f1], shape, &dev)?.to_dtype(dtype)?,
                g1.into_inner(),
                step_size,
                ls_func_evals,
            ))
        } else {
            Ok((
                Tensor::from_slice(&[f0], shape, &dev)?.to_dtype(dtype)?,
                g0.into_inner(),
                step_size,
                ls_func_evals,
            ))
        }
    }

    fn directional_evaluate(
        &mut self,
        mag: f64,
        direction: &Tensor,
    ) -> CResult<(Tensor, Tensor, f64)> {
        // need to cache the original result
        // Otherwise leads to drift over line search evals
        let original = self
            .vars
            .iter()
            .map(|v| v.as_tensor().copy())
            .collect::<CResult<Vec<Tensor>>>()?;

        add_grad(&mut self.vars, &(mag * direction)?)?;
        let loss = self.model.loss()?;
        let grad = flat_grads(&self.vars, &loss, self.params.weight_decay)?;
        let l2_reg = if let Some(wd) = self.params.weight_decay {
            0.5 * wd
                * self
                    .vars
                    .iter()
                    .map(|v| -> CResult<f64> {
                        v.as_tensor()
                            .sqr()?
                            .sum_all()?
                            .to_dtype(candle_core::DType::F64)?
                            .to_scalar::<f64>()
                    })
                    .sum::<CResult<f64>>()?
        } else {
            0.
        };

        set_vs(&mut self.vars, &original)?;
        // add_grad(&mut self.vars, &(-mag * direction)?)?;
        Ok((
            loss, //.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?
            grad, l2_reg,
        ))
    }

    fn l2_reg(&self) -> CResult<f64> {
        if let Some(wd) = self.params.weight_decay {
            Ok(0.5
                * wd
                * self
                    .vars
                    .iter()
                    .map(|v| -> CResult<f64> {
                        v.as_tensor()
                            .sqr()?
                            .sum_all()?
                            .to_dtype(candle_core::DType::F64)?
                            .to_scalar::<f64>()
                    })
                    .sum::<CResult<f64>>()?)
        } else {
            Ok(0.)
        }
    }
}

#[cfg(test)]
mod tests {
    // use candle_core::test_utils::{to_vec0_round, to_vec2_round};

    use crate::lbfgs::ParamsLBFGS;
    use crate::{LossOptimizer, Model};
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
    fn l2_test() -> Result<()> {
        let params = ParamsLBFGS {
            lr: 0.004,
            ..Default::default()
        };
        let (model, vars) = LinearModel::new()?;
        let lbfgs = Lbfgs::new(vars, params, model)?;
        let l2 = lbfgs.l2_reg()?;
        assert_approx_eq!(0.0, l2);

        let params = ParamsLBFGS {
            lr: 0.004,
            weight_decay: Some(1.0),
            ..Default::default()
        };
        let (model, vars) = LinearModel::new()?;
        let lbfgs = Lbfgs::new(vars, params, model)?;
        let l2 = lbfgs.l2_reg()?;
        assert_approx_eq!(7.0, l2); // 0.5 *(3^2 +1^2 + (-2)^2)
        Ok(())
    }
}
