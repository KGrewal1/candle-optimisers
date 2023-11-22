use candle_core::Result as CResult;
use candle_core::Tensor;
use candle_core::Var;
#[allow(dead_code)]
// pub fn strong_wolfe() {
//     todo!("Implement strong_wolfe");
// }
#[allow(dead_code)]
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

fn strong_wolfe<F: FnMut(&Tensor, f64, &Tensor) -> (f64, Tensor)>(
    mut obj_func: F,
    x: Tensor,
    mut t: f64,
    d: Tensor,
    f: f64,
    g: Tensor,
    gtd: f64,
    c1: f64,
    c2: f64,
    tolerance_change: f64,
    max_ls: usize,
) -> CResult<(f64, Tensor, f64, usize)> {
    // ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    let d_norm = &d
        .abs()?
        .max(0)?
        .to_dtype(candle_core::DType::F64)?
        .to_scalar::<f64>()?;
    // let g = g.clone();
    // evaluate objective and gradient using initial step
    let (mut f_new, g_new) = obj_func(&x, t, &d);
    let g_new = Var::from_tensor(&g_new)?;
    let mut ls_func_evals = 1;
    let mut gtd_new = (g_new.as_tensor() * &d)?
        .sum_all()?
        .to_dtype(candle_core::DType::F64)?
        .to_scalar::<f64>()?;

    // bracket an interval containing a point satisfying the Wolfe criteria
    let g_prev = Var::from_tensor(&g)?;
    let (mut t_prev, mut f_prev, mut gtd_prev) = (0., f, gtd);
    let mut done = false;
    let mut ls_iter = 0;
    // let mut bracket;
    // let mut bracket_f;
    // let mut bracket_g;
    let mut bracket_gtd;
    let (mut bracket, mut bracket_f, bracket_g) = loop {
        //while ls_iter < max_ls
        // check conditions
        if f_new > (f + c1 * t * gtd) || (ls_iter > 1 && f_new >= f_prev) {
            // bracket = [t_prev, t];
            // bracket_f = [f_prev, f_new];
            // bracket_g = [g_prev, g_new.clone()];
            bracket_gtd = [gtd_prev, gtd_new];
            break (
                [t_prev, t],
                [f_prev, f_new],
                [g_prev, Var::from_tensor(g_new.as_tensor())?],
            );
        }

        if gtd_new.abs() <= -c2 * gtd {
            // bracket = [t, t];
            // bracket_f = [f_new, f_new];
            // bracket_g = [g_new.clone(), g_new.clone()];
            done = true;
            bracket_gtd = [gtd_prev, gtd_new];
            break (
                [t, t],
                [f_new, f_new],
                [
                    Var::from_tensor(g_new.as_tensor())?,
                    Var::from_tensor(g_new.as_tensor())?,
                ],
            );
        }

        if gtd_new >= 0. {
            // bracket = [t_prev, t];
            // bracket_f = [f_prev, f_new];
            // bracket_g = [g_prev, g_new.clone()];
            bracket_gtd = [gtd_prev, gtd_new];
            break (
                [t_prev, t],
                [f_prev, f_new],
                [g_prev, Var::from_tensor(g_new.as_tensor())?],
            );
        }

        // interpolate
        let min_step = t + 0.01 * (t - t_prev);
        let max_step = t * 10.;
        let tmp = t;
        t = cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            Some((min_step, max_step)),
        );

        // next step
        t_prev = tmp;
        f_prev = f_new;
        g_prev.set(g_new.as_tensor())?;
        gtd_prev = gtd_new;
        // assign to temp vars: (f_new, g_new) = obj_func(&x, t, &d);
        let (next_f, next_g) = obj_func(&x, t, &d);
        // overwrite
        f_new = next_f;
        g_new.set(&next_g)?;
        //
        ls_func_evals += 1;
        gtd_new = (g_new.as_tensor() * &d)?
            .sum_all()?
            .to_dtype(candle_core::DType::F64)?
            .to_scalar::<f64>()?;
        ls_iter += 1;

        // reached max number of iterations?
        if ls_iter == max_ls {
            // let bracket = [0., t];
            // let bracket_f = [f, f_new];
            // let bracket_g = [g, g_new];
            bracket_gtd = [gtd_prev, gtd_new];
            break (
                [0., t],
                [f, f_new],
                [Var::from_tensor(&g)?, Var::from_tensor(g_new.as_tensor())?],
            );
        }
    };

    // zoom phase: we now have a point satisfying the criteria, or
    // a bracket around it. We refine the bracket until we find the
    // exact point satisfying the criteria
    let mut insuf_progress = false;
    // find high and low points in bracket
    let (mut low_pos, mut high_pos) = if bracket_f[0] <= bracket_f[1] {
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
        t = cubic_interpolate(
            bracket[0],
            bracket_f[0],
            bracket_gtd[0],
            bracket[1],
            bracket_f[1],
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
        if (max_bracket - t).min(t - min_bracket) < eps {
            // interpolation close to boundary
            if insuf_progress || t >= max_bracket || t <= min_bracket {
                // evaluate at 0.1 away from boundary
                if (t - max_bracket).abs() < (t - min_bracket).abs() {
                    t = max_bracket - eps
                } else {
                    t = min_bracket + eps
                }
                insuf_progress = false
            } else {
                insuf_progress = true
            }
        } else {
            insuf_progress = false
        }

        // Evaluate new point
        // assign to temp vars: (f_new, g_new) = obj_func(&x, t, &d);
        let (next_f, next_g) = obj_func(&x, t, &d);
        // overwrite
        f_new = next_f;
        g_new.set(&next_g)?;
        ls_func_evals += 1;
        gtd_new = (g_new.as_tensor() * &d)?
            .sum_all()?
            .to_dtype(candle_core::DType::F64)?
            .to_scalar::<f64>()?;
        ls_iter += 1;

        if f_new > (f + c1 * t * gtd) || f_new >= bracket_f[low_pos] {
            // Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t;
            bracket_f[high_pos] = f_new;
            bracket_g[high_pos].set(g_new.as_tensor())?; //.clone()
            bracket_gtd[high_pos] = gtd_new;
            (low_pos, high_pos) = if bracket_f[0] <= bracket_f[1] {
                (0, 1)
            } else {
                (1, 0)
            };
        } else {
            if gtd_new.abs() <= -c2 * gtd {
                // Wolfe conditions satisfied
                done = true
            } else if gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0. {
                // old high becomes new low
                bracket[high_pos] = bracket[low_pos];
                bracket_f[high_pos] = bracket_f[low_pos];
                bracket_g[high_pos].set(bracket_g[low_pos].as_tensor())?;
                bracket_gtd[high_pos] = bracket_gtd[low_pos]
            }

            // new point becomes new low
            bracket[low_pos] = t;
            bracket_f[low_pos] = f_new;
            bracket_g[low_pos].set(g_new.as_tensor())?;
            bracket_gtd[low_pos] = gtd_new;
        }
    }

    // return stuff
    t = bracket[low_pos];
    f_new = bracket_f[low_pos];
    // g_new = ;
    // f_new, g_new, t, ls_func_evals;
    Ok((
        f_new,
        bracket_g[low_pos].clone().into_inner(),
        t,
        ls_func_evals,
    ))
}
