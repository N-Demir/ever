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

import jax
import jax.numpy as jnp

import numpy as np
tiny_val = np.float32(np.finfo(np.float32).tiny)
min_val = np.float32(np.finfo(np.float32).min)
max_val = np.float32(np.finfo(np.float32).max)

@jax.custom_jvp
def plus_eps(x):
  return jnp.where(
      jnp.abs(x) < tiny_val, tiny_val, jnp.nextafter(jnp.float32(x), jnp.inf)
  )


@jax.custom_jvp
def minus_eps(x):
  return jnp.where(
      jnp.abs(x) < tiny_val, -tiny_val, jnp.nextafter(jnp.float32(x), -jnp.inf)
  )


@plus_eps.defjvp
def plus_eps_jvp(primals, tangents):
  """Make plus_eps()'s gradient a no-op (nextafter's gradient is undefined)."""
  return plus_eps(*primals), tangents[0]


@minus_eps.defjvp
def minus_eps_jvp(primals, tangents):
  """Make minus_eps()'s gradient a no-op (nextafter's gradient is undefined)."""
  return minus_eps(*primals), tangents[0]

def generate_clip_nograd_fn(a_min, a_max):
  """Generates a function that clips to [a_min, a_max] with no grad effects."""

  @jax.custom_jvp
  def clip_nograd(a):
    """Clamps `a` from above and below."""
    return jnp.clip(a, a_min, a_max)

  @clip_nograd.defjvp
  def clip_nograd_jvp(primals, tangents):
    """Override clips()'s gradient to be a no-op."""
    return clip_nograd(primals[0]), tangents[0]

  return clip_nograd


def safe_trig_helper(x, fn, t=100 * jnp.pi):
  """Helper function used by safe_cos/safe_sin: mods x before sin()/cos()."""
  return fn(jnp.nan_to_num(jnp.where(jnp.abs(x) < t, x, x % t)))


def safe_cos(x):
  """jnp.cos() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, jnp.cos)


def safe_sin(x):
  """jnp.sin() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, jnp.sin)


@jax.custom_vjp
def safe_arctan2(x1, x2):
  return safe_arctan2_fwd(x1, x2)[0]


def safe_arctan2_fwd(x1, x2):
  return jnp.arctan2(x1, x2), (x1, x2)


def safe_arctan2_bwd(res, g):
  x1, x2 = res
  denom = remove_zero(x1**2 + x2**2)
  d1 = g * (x2 / denom)
  d2 = g * (-x1 / denom)
  return d1, d2


safe_arctan2.defvjp(safe_arctan2_fwd, safe_arctan2_bwd)


def generate_clip_nograd_fn(a_min, a_max):
  """Generates a function that clips to [a_min, a_max] with no grad effects."""

  @jax.custom_jvp
  def clip_nograd(a):
    """Clamps `a` from above and below."""
    return jnp.clip(a, a_min, a_max)

  @clip_nograd.defjvp
  def clip_nograd_jvp(primals, tangents):
    """Override clips()'s gradient to be a no-op."""
    return clip_nograd(primals[0]), tangents[0]

  return clip_nograd


clip_finite_nograd = generate_clip_nograd_fn(min_val, max_val)

clip_pos_finite_nograd = generate_clip_nograd_fn(tiny_val, max_val)


def clip_pos(x):
  """Clamps `x` from below to be positive."""
  return jnp.maximum(tiny_val, x)


def safe_sign(x):
  """jnp.sign(x) except x=0 is assumed to have a sign of +1, not 0."""
  return jnp.where(x < 0, -1, +1)


def remove_zero(x):
  """Shifts `x` away from 0."""
  return jnp.where(jnp.abs(x) < tiny_val, tiny_val, x)


def clip_finite(x):
  return jnp.clip(x, min_val, max_val)


@jax.custom_vjp
def safe_div(n, d):
  """Divide `n` by `d` but the value and gradient never nan out."""
  return safe_div_fwd(n, d)[0]


def safe_div_fwd(n, d):
  r = jnp.clip(n / remove_zero(d), min_val, max_val)
  return jnp.where(jnp.abs(d) < tiny_val, 0, r), (d, r)


def safe_div_bwd(res, g):
  d, r = res
  dn = jnp.clip(g / remove_zero(d), min_val, max_val)
  dd = jnp.clip(-g * r / remove_zero(d), min_val, max_val)
  return dn, dd


safe_div.defvjp(safe_div_fwd, safe_div_bwd)


def generate_safe_fn(fn, grad_fn, x_range):
  """Generate's a `safe` fn() where inputs are clipped in fwd and bwd passes."""

  @jax.custom_jvp
  def safe_fn(x):
    """fn() with clipped inputs."""
    return fn(jnp.clip(x, *x_range))

  @safe_fn.defjvp
  def safe_fn_jvp(primals, tangents):
    """Backpropagate using the gradient and clipped inputs."""
    (x,) = primals
    (x_dot,) = tangents
    y = safe_fn(x)
    y_dot = grad_fn(jnp.clip(x, *x_range), y, x_dot)
    return y, y_dot

  return safe_fn


# These safe_* functions need to be wrapped in no-op function definitions for
# gin to recognize them, otherwise they could just be calls to generate_safe_fn.


def safe_log(x):
  return generate_safe_fn(
      jnp.log,
      lambda x, _, x_dot: x_dot / x,
      (tiny_val, max_val),
  )(x)


def safe_exp(x):
  return generate_safe_fn(
      jnp.exp,
      lambda _, y, x_dot: y * x_dot,
      (min_val, np.nextafter(np.log(max_val), np.float32(0))),
  )(x)


def safe_sqrt(x):
  return generate_safe_fn(
      jnp.sqrt,
      lambda x, _, x_dot: 0.5 * x_dot / jnp.sqrt(jnp.maximum(tiny_val, x)),
      (0, max_val),
  )(x)


def inverse_sigmoid(x):
  return jnp.log(x / (1-x))
