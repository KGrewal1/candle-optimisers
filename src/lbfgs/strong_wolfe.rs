use candle_core::Result as CResult;
use candle_core::{Tensor, Var};

use crate::Model;

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
        // let x = &self.vars;
        let d_norm = &direction
            .abs()?
            .max(0)?
            .to_dtype(candle_core::DType::F64)?
            .to_scalar::<f64>()?;

        // evaluate objective and gradient using initial step
        let (f_new, g_new) = self.directional_evaluate(step_size, direction)?; //obj_func(x, t, &d);
        let g_new = Var::from_tensor(&g_new)?;
        let f_new = Var::from_tensor(&f_new)?;
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
        let f_prev = Var::from_tensor(loss)?;
        let (mut t_prev, mut gtd_prev) = (0., directional_grad);
        let mut done = false;
        let mut ls_iter = 0;

        let mut bracket_gtd;
        let bracket_f; // bracket_f,
        let (mut bracket, bracket_g) = loop {
            //while ls_iter < max_ls
            // check conditions
            if f_new
                .to_dtype(candle_core::DType::F64)?
                .to_scalar::<f64>()?
                > (loss.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?
                    + c1 * step_size * directional_grad)
                || (ls_iter > 1
                    && f_new
                        .to_dtype(candle_core::DType::F64)?
                        .to_scalar::<f64>()?
                        >= f_prev
                            .to_dtype(candle_core::DType::F64)?
                            .to_scalar::<f64>()?)
            {
                bracket_gtd = [gtd_prev, gtd_new];
                bracket_f = [f_prev, Var::from_tensor(f_new.as_tensor())?]; //.clone()
                break (
                    [t_prev, step_size],
                    //[f_prev, f_new.clone()],
                    [g_prev, Var::from_tensor(g_new.as_tensor())?],
                );
            }

            if gtd_new.abs() <= -c2 * directional_grad {
                done = true;
                bracket_gtd = [gtd_prev, gtd_new];
                bracket_f = [
                    Var::from_tensor(f_new.as_tensor())?,
                    Var::from_tensor(f_new.as_tensor())?,
                ]; //.clone()
                break (
                    [step_size, step_size],
                    // [f_new.clone(), f_new.clone()],
                    [
                        Var::from_tensor(g_new.as_tensor())?,
                        Var::from_tensor(g_new.as_tensor())?,
                    ],
                );
            }

            if gtd_new >= 0. {
                bracket_gtd = [gtd_prev, gtd_new];
                bracket_f = [f_prev, Var::from_tensor(f_new.as_tensor())?];
                break (
                    [t_prev, step_size],
                    // [f_prev, f_new.clone()],
                    [g_prev, Var::from_tensor(g_new.as_tensor())?],
                );
            }

            // interpolate
            let min_step = step_size + 0.01 * (step_size - t_prev);
            let max_step = step_size * 10.;
            let tmp = step_size;
            step_size = cubic_interpolate(
                t_prev,
                f_prev
                    .to_dtype(candle_core::DType::F64)?
                    .to_scalar::<f64>()?,
                gtd_prev,
                step_size,
                f_new
                    .to_dtype(candle_core::DType::F64)?
                    .to_scalar::<f64>()?,
                gtd_new,
                Some((min_step, max_step)),
            );

            // next step
            t_prev = tmp;
            f_prev.set(f_new.as_tensor())?;
            g_prev.set(g_new.as_tensor())?;
            gtd_prev = gtd_new;
            // assign to temp vars: (f_new, g_new) = obj_func(&x, t, &d);
            let (next_f, next_g) = self.directional_evaluate(step_size, direction)?;
            // overwrite
            f_new.set(&next_f)?;
            g_new.set(&next_g)?;
            //
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
                bracket_f = [
                    Var::from_tensor(loss)?,
                    Var::from_tensor(f_new.as_tensor())?,
                ]; //.clone()
                break (
                    [0., step_size],
                    // [Var::from_tensor(loss)?, f_new.clone()],
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
        let (mut low_pos, mut high_pos) = if bracket_f[0]
            .to_dtype(candle_core::DType::F64)?
            .to_scalar::<f64>()?
            <= bracket_f[1]
                .to_dtype(candle_core::DType::F64)?
                .to_scalar::<f64>()?
        {
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
                bracket_f[0]
                    .to_dtype(candle_core::DType::F64)?
                    .to_scalar::<f64>()?,
                bracket_gtd[0],
                bracket[1],
                bracket_f[1]
                    .to_dtype(candle_core::DType::F64)?
                    .to_scalar::<f64>()?,
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
            // assign to temp vars: (f_new, g_new) = obj_func(&x, t, &d);
            let (next_f, next_g) = self.directional_evaluate(step_size, direction)?;
            // overwrite
            f_new.set(&next_f)?;
            g_new.set(&next_g)?;
            ls_func_evals += 1;

            gtd_new = g_new
                .unsqueeze(0)?
                .matmul(&(direction.unsqueeze(1)?))?
                .to_dtype(candle_core::DType::F64)?
                .squeeze(1)?
                .squeeze(0)?
                .to_scalar::<f64>()?;
            ls_iter += 1;

            if f_new
                .to_dtype(candle_core::DType::F64)?
                .to_scalar::<f64>()?
                > (loss.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?
                    + c1 * step_size * directional_grad)
                || f_new
                    .to_dtype(candle_core::DType::F64)?
                    .to_scalar::<f64>()?
                    >= bracket_f[low_pos]
                        .to_dtype(candle_core::DType::F64)?
                        .to_scalar::<f64>()?
            {
                // Armijo condition not satisfied or not lower than lowest point
                bracket[high_pos] = step_size;
                bracket_f[high_pos].set(&f_new)?;
                bracket_g[high_pos].set(g_new.as_tensor())?; //.clone()
                bracket_gtd[high_pos] = gtd_new;
                (low_pos, high_pos) = if bracket_f[0]
                    .to_dtype(candle_core::DType::F64)?
                    .to_scalar::<f64>()?
                    <= bracket_f[1]
                        .to_dtype(candle_core::DType::F64)?
                        .to_scalar::<f64>()?
                {
                    (0, 1)
                } else {
                    (1, 0)
                };
            } else {
                if gtd_new.abs() <= -c2 * directional_grad {
                    // Wolfe conditions satisfied
                    done = true;
                } else if gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0. {
                    // old high becomes new low
                    bracket[high_pos] = bracket[low_pos];
                    bracket_f[high_pos].set(bracket_f[low_pos].as_tensor())?;
                    bracket_g[high_pos].set(bracket_g[low_pos].as_tensor())?;
                    bracket_gtd[high_pos] = bracket_gtd[low_pos];
                }

                // new point becomes new low
                bracket[low_pos] = step_size;
                bracket_f[low_pos].set(f_new.as_tensor())?;
                bracket_g[low_pos].set(g_new.as_tensor())?;
                bracket_gtd[low_pos] = gtd_new;
            }
        }

        // return new value, new grad, line-search value, nb of function evals
        step_size = bracket[low_pos];
        // f_new = bracket_f[low_pos];
        // g_new = ;
        // f_new, g_new, t, ls_func_evals;
        let [g0, g1] = bracket_g;
        let [f0, f1] = bracket_f;
        if low_pos == 1 {
            // if b is the lower value set a to b, else a should be returned
            Ok((f1.into_inner(), g1.into_inner(), step_size, ls_func_evals))
        } else {
            Ok((f0.into_inner(), g0.into_inner(), step_size, ls_func_evals))
        }
    }
}

impl<M: Model> Lbfgs<M> {
    fn directional_evaluate(&mut self, mag: f64, direction: &Tensor) -> CResult<(Tensor, Tensor)> {
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
        set_vs(&mut self.vars, &original)?;
        // add_grad(&mut self.vars, &(-mag * direction)?)?;
        Ok((
            loss, //.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?
            grad,
        ))
    }
}
