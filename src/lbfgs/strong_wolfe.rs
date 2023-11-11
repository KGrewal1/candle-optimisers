#[allow(dead_code)]
pub fn strong_wolfe() {
    todo!("Implement strong_wolfe");
}

#[allow(dead_code)]
fn cubic_interpolate(
    x1: f64,
    f1: f64,
    g1: f64,
    x2: f64,
    f2: f64,
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

// fn strong_wolfe(obj_func,
//                   x,
//                   t,
//                   d,
//                   f,
//                   g,
//                   gtd,
//                   c1: f64,
//                   c2: f64,
//                   tolerance_change: f64,
//                   max_ls: usize){
//     // ported from https://github.com/torch/optim/blob/master/lswolfe.lua
//     d_norm = d.abs().max()
//     g = g.clone(memory_format=torch.contiguous_format)
//     // evaluate objective and gradient using initial step
//     f_new, g_new = obj_func(x, t, d)
//     ls_func_evals = 1
//     gtd_new = g_new.dot(d)

//     // bracket an interval containing a point satisfying the Wolfe criteria
//     t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
//     done = False
//     ls_iter = 0
//     while ls_iter < max_ls:
//         // check conditions
//         if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
//             bracket = [t_prev, t]
//             bracket_f = [f_prev, f_new]
//             bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
//             bracket_gtd = [gtd_prev, gtd_new]
//             break

//         if abs(gtd_new) <= -c2 * gtd:
//             bracket = [t]
//             bracket_f = [f_new]
//             bracket_g = [g_new]
//             done = True
//             break

//         if gtd_new >= 0:
//             bracket = [t_prev, t]
//             bracket_f = [f_prev, f_new]
//             bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
//             bracket_gtd = [gtd_prev, gtd_new]
//             break

//         // interpolate
//         min_step = t + 0.01 * (t - t_prev)
//         max_step = t * 10
//         tmp = t
//         t = _cubic_interpolate(
//             t_prev,
//             f_prev,
//             gtd_prev,
//             t,
//             f_new,
//             gtd_new,
//             bounds=(min_step, max_step))

//         // next step
//         t_prev = tmp
//         f_prev = f_new
//         g_prev = g_new.clone(memory_format=torch.contiguous_format)
//         gtd_prev = gtd_new
//         f_new, g_new = obj_func(x, t, d)
//         ls_func_evals += 1
//         gtd_new = g_new.dot(d)
//         ls_iter += 1

//     // reached max number of iterations?
//     if ls_iter == max_ls:
//         bracket = [0, t]
//         bracket_f = [f, f_new]
//         bracket_g = [g, g_new]

//     // zoom phase: we now have a point satisfying the criteria, or
//     // a bracket around it. We refine the bracket until we find the
//     // exact point satisfying the criteria
//     insuf_progress = False
//     // find high and low points in bracket
//     low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
//     while not done and ls_iter < max_ls:
//         // line-search bracket is so small
//         if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
//             break

//         // compute new trial value
//         t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
//                                bracket[1], bracket_f[1], bracket_gtd[1])

//         // test that we are making sufficient progress:
//         // in case `t` is so close to boundary, we mark that we are making
//         // insufficient progress, and if
//         //   + we have made insufficient progress in the last step, or
//         //   + `t` is at one of the boundary,
//         // we will move `t` to a position which is `0.1 * len(bracket)`
//         // away from the nearest boundary point.
//         eps = 0.1 * (max(bracket) - min(bracket))
//         if min(max(bracket) - t, t - min(bracket)) < eps:
//             // interpolation close to boundary
//             if insuf_progress or t >= max(bracket) or t <= min(bracket):
//                 // evaluate at 0.1 away from boundary
//                 if abs(t - max(bracket)) < abs(t - min(bracket)):
//                     t = max(bracket) - eps
//                 else:
//                     t = min(bracket) + eps
//                 insuf_progress = False
//             else:
//                 insuf_progress = True
//         else:
//             insuf_progress = False

//         // Evaluate new point
//         f_new, g_new = obj_func(x, t, d)
//         ls_func_evals += 1
//         gtd_new = g_new.dot(d)
//         ls_iter += 1

//         if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
//             // Armijo condition not satisfied or not lower than lowest point
//             bracket[high_pos] = t
//             bracket_f[high_pos] = f_new
//             bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
//             bracket_gtd[high_pos] = gtd_new
//             low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
//         else:
//             if abs(gtd_new) <= -c2 * gtd:
//                 // Wolfe conditions satisfied
//                 done = True
//             elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
//                 // old high becomes new low
//                 bracket[high_pos] = bracket[low_pos]
//                 bracket_f[high_pos] = bracket_f[low_pos]
//                 bracket_g[high_pos] = bracket_g[low_pos]
//                 bracket_gtd[high_pos] = bracket_gtd[low_pos]

//             // new point becomes new low
//             bracket[low_pos] = t
//             bracket_f[low_pos] = f_new
//             bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
//             bracket_gtd[low_pos] = gtd_new

//     // return stuff
//     t = bracket[low_pos]
//     f_new = bracket_f[low_pos]
//     g_new = bracket_g[low_pos]
//     return f_new, g_new, t, ls_func_evals
// }
