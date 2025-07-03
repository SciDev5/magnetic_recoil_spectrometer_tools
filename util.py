""" """

import typing as ty
import time

import numpy as np
import numpy.typing as npt

########################################################################
# ==== CALCULUS


def d_(x: npt.NDArray) -> npt.NDArray:
    return x[1] - x[0]


def dsolid_dangle(a: npt.NDArray) -> npt.NDArray:
    return 2 * np.pi * np.sin(a)


def dsolid_dspherical(a: npt.NDArray) -> npt.NDArray:
    return np.sin(a)


########################################################################
# ==== RANDOMIZATION

rng = np.random.default_rng()


# [i], [j], [i, j, ..] -> [i, ..]
def interp2(x: npt.ArrayLike, xp: npt.ArrayLike, fp: npt.ArrayLike) -> npt.NDArray:
    # https://stackoverflow.com/questions/43772218/fastest-way-to-use-numpy-interp-on-a-2-d-array
    x = np.array(x)
    xp = np.array(xp)
    fp = np.array(fp)
    i = np.arange(x.size)
    j = np.clip(np.searchsorted(xp, x) - 1, a_min=0, a_max=xp.size - 2)
    d = np.clip((x - xp[j]) / (xp[j + 1] - xp[j]), 0, 1)
    if len(fp.shape) > 2:
        axes = np.array(range(len(fp.shape) - 1))
        axes[[0, -1]] = axes[[-1, 0]]
        return np.transpose(
            (1 - d) * np.transpose(fp[i, j], axes=axes)
            + np.transpose(fp[i, j + 1], axes=axes) * d,
            axes=axes,
        )
    else:
        return (1 - d) * fp[i, j] + fp[i, j + 1] * d


# [i], [j], [j, ..] -> [i, ..]
def interpx(x: npt.ArrayLike, xp: npt.ArrayLike, fp: npt.ArrayLike) -> npt.NDArray:
    # https://stackoverflow.com/questions/43772218/fastest-way-to-use-numpy-interp-on-a-2-d-array
    x = np.array(x)
    xp = np.array(xp)
    fp = np.array(fp)
    j = np.clip(np.searchsorted(xp, x) - 1, a_min=0, a_max=xp.size - 2)
    d = np.clip((x - xp[j]) / (xp[j + 1] - xp[j]), 0, 1)
    if len(fp.shape) > 1:
        axes = np.array(range(len(fp.shape)))
        axes[[0, -1]] = axes[[-1, 0]]
        return np.transpose(
            (1 - d) * np.transpose(fp[j], axes=axes)
            + np.transpose(fp[j + 1], axes=axes) * d,
            axes=axes,
        )
    else:
        return (1 - d) * fp[j] + fp[j + 1] * d


# [count, dist_len]
def sample_dists(
    dists: npt.NDArray, out_range: npt.NDArray | None = None
) -> npt.NDArray:
    if out_range is None:
        out_range = np.arange(dists.shape[1])
    return np.array([np.interp(rng.random(), dist, out_range) for dist in dists])


def normalize_axis(x_in: npt.ArrayLike, axis: int) -> npt.NDArray:
    x = np.array(x_in)
    axes = np.array(range(len(x.shape)))
    axes[[0, axis]] = axes[[axis, 0]]

    x = x.transpose(axes)
    x -= x[0]
    return np.transpose(x / x[-1], axes=axes)


Distr_1_2 = tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]


def prep_distr_1_2(
    dist: npt.NDArray,  # [arg_axis, distr_axis_0, distr_axis_1]
    arg_range: npt.NDArray,  # [arg_axis]
    distr_axis_0_range: npt.NDArray,  # [distr_axis_0]
    distr_axis_1_range: npt.NDArray,  # [distr_axis_1]
) -> Distr_1_2:
    dist_0 = normalize_axis(np.cumsum(dist.sum(axis=(2,)), axis=1), axis=1)
    dist_1 = normalize_axis(dist.cumsum(axis=2), axis=2)
    return dist_0, dist_1, arg_range, distr_axis_0_range, distr_axis_1_range


def sample_distr_1_2(
    count: int,
    dist_1_2: Distr_1_2,
    argument: float,
) -> tuple[npt.NDArray, npt.NDArray]:
    dist_0, dist_1, arg_range, distr_axis_0_range, distr_axis_1_range = dist_1_2

    dist_0_interpolated: npt.NDArray = interpx(
        [argument],
        arg_range,
        dist_0,
    )[0]
    x0 = np.interp(
        rng.random(count),
        dist_0_interpolated,
        distr_axis_0_range,
    )
    dist_1_interpolated = interpx(
        x0,
        distr_axis_0_range,
        interpx([argument], arg_range, dist_1)[0],
    )
    x1 = sample_dists(dist_1_interpolated, out_range=distr_axis_1_range)
    return x0, x1


Distr_1_1 = tuple[npt.NDArray, npt.NDArray, npt.NDArray]


def prep_distr_1_1(
    dist: npt.NDArray,  # [arg_axis, distr_axis_0]
    arg_range: npt.NDArray,  # [arg_axis]
    distr_axis_0_range: npt.NDArray,  # [distr_axis_0]
) -> Distr_1_1:
    return (
        normalize_axis(np.cumsum(dist, axis=1), axis=1),
        arg_range,
        distr_axis_0_range,
    )


def sample_distr_1_1(
    count: int,
    dist: Distr_1_1,
    argument: float,
) -> npt.NDArray:
    dist_0, arg_range, distr_axis_0_range = dist
    dist_interpolated: npt.NDArray = interpx(
        [argument],
        arg_range,
        dist_0,
    )[0]
    x0 = np.interp(
        rng.random(count),
        dist_interpolated,
        distr_axis_0_range,
    )
    return x0


def random_partition(rel_probabilities: npt.ArrayLike, count: int) -> npt.NDArray:
    out = []
    rel_probabilities = np.array(rel_probabilities)
    rel_probabilities /= rel_probabilities.sum()
    probability_accum = 0
    for probability in rel_probabilities[:-1]:
        p = np.clip(probability / (1 - probability_accum), 0, 1)
        take = rng.binomial(count, p)
        probability_accum += probability
        count -= take
        out.append(take)
    out.append(count)
    return np.array(out)


########################################################################
# ==== TIMING


def begin_timer(
    text: str,
) -> ty.Callable[[float], None]:
    t_start = time.time()
    done = [False]
    did_tick = [False]

    def fmt_duration(duration_seconds: float, decimals=1) -> str:
        if duration_seconds < 0.5:
            return f"{np.round(duration_seconds*1e3,decimals)}ms"
        if duration_seconds < 60 * 2:
            return f"{np.round(duration_seconds,decimals)}s"
        if duration_seconds < 60 * 60 * 2:
            return f"{np.round(duration_seconds/60,decimals)}min"
        if duration_seconds < 60 * 60 * 24 * 2:
            return f"{np.round(duration_seconds/60/60,decimals)}hr"
        if duration_seconds < 60 * 60 * 24 * 100:
            return f"{np.round(duration_seconds/60/60/24,decimals)}dy"
        return f"{np.round(duration_seconds/60/60/24/365.24,decimals)}yrs"

    def print_state(complete: float):
        elapsed = time.time() - t_start
        total_time_est = np.round(elapsed / complete, decimals=1)
        show_date = total_time_est > 60 * 60 * 12  # more than 12 hours
        time_fmt = "%Y-%m-%d %H:%M:%S" if show_date else "%H:%M:%S"
        now_time_fmt = time.strftime(time_fmt, time.localtime(time.time()))
        end_time_fmt = time.strftime(time_fmt, time.localtime(t_start + total_time_est))
        if complete == 1:
            print(
                f"[{now_time_fmt}]  {text} complete in {fmt_duration(elapsed, decimals=3)}"
            )
        else:
            print(
                f"[{now_time_fmt}]  {text}:  {np.round(complete * 100, decimals=2)}%  {fmt_duration(elapsed)}/{fmt_duration(total_time_est)}  ETA: {end_time_fmt} [T-{fmt_duration(total_time_est-elapsed)}]"
            )

    t_next_update_delay = [1]
    t_next_update = [time.time() + t_next_update_delay[0]]

    def tick(complete: float):
        if done[0] or not (complete >= 0):
            return
        if complete >= 1 and did_tick[0]:
            done[0] = True
            print_state(1)
        if time.time() < t_next_update[0]:
            return

        print_state(complete)
        did_tick[0] = True
        t_next_update_delay[0] *= 1.5
        t_next_update[0] = time.time() + t_next_update_delay[0]

    return tick
