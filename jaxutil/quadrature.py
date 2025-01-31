# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import math

import jax
import jax.numpy as jnp
from jaxutil import safe_math


def assert_valid_stepfun(t, y):
  """Assert that step function (t, y) has a valid shape."""
  if t.shape[-1] != y.shape[-1] + 1:
    raise ValueError(
        f'Invalid shapes ({t.shape}, {y.shape}) for a step function.'
    )

@jax.jit
def lossfun_distortion(t, w):
  """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
  assert_valid_stepfun(t, w)

  # The loss incurred between all pairs of intervals.
  ut = (t[Ellipsis, 1:] + t[Ellipsis, :-1]) / 2
  dut = jnp.abs(ut[Ellipsis, :, None] - ut[Ellipsis, None, :])
  loss_inter = jnp.sum(w * jnp.sum(w[Ellipsis, None, :] * dut, axis=-1), axis=-1)

  # The loss incurred within each individual interval with itself.
  loss_intra = jnp.sum(w**2 * jnp.diff(t), axis=-1) / 3

  return loss_inter + loss_intra


def log1mexp(x):
    """Accurate computation of log(1 - exp(-x)) for x > 0, thanks watsondaniel."""
    # https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    return safe_math.safe_log(1-jnp.exp(-x))


def compute_alpha_weights_helper(density_delta):
    log_trans = -jnp.concatenate(
        [
            jnp.zeros_like(density_delta[..., :1]),
            jnp.cumsum(density_delta[..., :-1], axis=-1),
        ],
        axis=-1,
    )

    log_weights = log1mexp(density_delta) + log_trans
    weights = jnp.exp(log_weights)
    return weights

def render_quadrature(tdist, query_fn, return_extras=False):
    """Numerical quadrature rendering of a set of colored Gaussians."""
    t_avg = 0.5 * (tdist[..., 1:] + tdist[..., :-1])
    t_delta = jnp.diff(tdist)
    total_density, avg_colors = query_fn(t_avg)
    weights = compute_alpha_weights_helper(total_density * t_delta)
    dist_loss = lossfun_distortion(tdist, weights)
    rendered_color = jnp.sum(
        weights[..., None] * avg_colors, axis=-2
    )  # Assuming the bg color is 0.
    alpha = jnp.sum(weights, axis=-1).reshape(-1, 1)
    rendered_color = jnp.concatenate([
        rendered_color.reshape(-1, 3), alpha, dist_loss.reshape(-1, 1)
    ], axis=1)

    if return_extras:
        return rendered_color, {
            "tdist": tdist,
            "avg_colors": avg_colors,
            "weights": weights,
            "total_density": jnp.sum(total_density*t_delta, axis=-1),
        }
    else:
        return rendered_color

