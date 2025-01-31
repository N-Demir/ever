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

import numpy as np
from utils import stepfun
from icecream import ic

def rotation_about_axis(degrees, axis=0):
  """Creates rotation matrix about one of the coordinate axes."""
  radians = degrees / 180.0 * np.pi
  rot2x2 = np.array(
      [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]
  )
  r = np.eye(3)
  r[1:3, 1:3] = rot2x2
  r = np.roll(np.roll(r, axis, axis=0), axis, axis=1)
  p = np.eye(4)
  p[:3, :3] = r
  return p


def normalize(x):
  """Normalization helper function."""
  return x / np.linalg.norm(x)


def focus_point_fn(poses, xnp = np):
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = xnp.eye(3) - directions * xnp.transpose(directions, [0, 2, 1])
  mt_m = xnp.transpose(m, [0, 2, 1]) @ m
  focus_pt = xnp.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt

def viewmatrix(
    lookdir,
    up,
    position,
    lock_up = False,
):
  """Construct lookat view matrix."""
  orthogonal_dir = lambda a, b: normalize(np.cross(a, b))
  vecs = [None, normalize(up), normalize(lookdir)]
  # x-axis is always the normalized cross product of `lookdir` and `up`.
  vecs[0] = orthogonal_dir(vecs[1], vecs[2])
  # Default is to lock `lookdir` vector, if lock_up is True lock `up` instead.
  ax = 2 if lock_up else 1
  # Set the not-locked axis to be orthogonal to the other two.
  vecs[ax] = orthogonal_dir(vecs[(ax + 1) % 3], vecs[(ax + 2) % 3])
  m = np.stack(vecs + [position], axis=1)
  return m


def generate_ellipse_path(
    poses,
    n_frames = 120,
    const_speed = True,
    z_variation = 0.0,
    z_phase = 0.0,
    rad_mult_min = 1.0,
    rad_mult_max = 1.0,
    render_rotate_xaxis = 0.0,
    render_rotate_yaxis = 0.0,
    use_avg_z_height = True,
    z_height_percentile = None,
    lock_up = False,
):
  """Generate an elliptical render path based on the given poses."""
  # Calculate the focal point for the path (cameras point toward this).
  center = focus_point_fn(poses)
  # Default path height sits at z=0 (in middle of zero-mean capture pattern).
  xy_offset = center[:2]

  # Calculate lengths for ellipse axes based on input camera positions.
  xy_radii = np.percentile(np.abs(poses[:, :2, 3] - xy_offset), 90, axis=0)
  # Use ellipse that is symmetric about the focal point in xy.
  xy_low = xy_offset - xy_radii
  xy_high = xy_offset + xy_radii

  # Optional height variation, need not be symmetric.
  z_min = np.percentile((poses[:, 2, 3]), 10, axis=0)
  z_max = np.percentile((poses[:, 2, 3]), 90, axis=0)
  # ic(z_min, z_max)
  if use_avg_z_height or z_height_percentile is not None:
    # Center the path vertically around the average camera height, good for
    # datasets recentered by transform_poses_focus function.
    if z_height_percentile is None:
      z_init = poses[:, 2, 3].mean(axis=0)
    else:
      z_init = np.percentile(poses[:, 2, 3], z_height_percentile, axis=0)
  else:
    # Center the path at zero, good for datasets recentered by
    # transform_poses_pca function.
    z_init = 0
  z_low = z_init + z_variation * (z_min - z_init)
  z_high = z_init + z_variation * (z_max - z_init)

  xyz_low = np.array([*xy_low, z_low])
  xyz_high = np.array([*xy_high, z_high])

  def get_positions(theta):
    # Interpolate between bounds with trig functions to get ellipse in x-y.
    # Optionally also interpolate in z to change camera height along path.
    t_x = np.cos(theta) * 0.5 + 0.5
    t_y = np.sin(theta) * 0.5 + 0.5
    t_z = np.cos(theta + 2 * np.pi * z_phase) * 0.5 + 0.5
    t_xyz = np.stack([t_x, t_y, t_z], axis=-1)
    positions = xyz_low + t_xyz * (xyz_high - xyz_low)
    # Interpolate between min and max radius multipliers so the camera zooms in
    # and out of the scene center.
    t = np.sin(theta) * 0.5 + 0.5
    rad_mult = rad_mult_min + (rad_mult_max - rad_mult_min) * t
    positions = center + (positions - center) * rad_mult[:, None]
    return positions

  theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
  positions = get_positions(theta)

  if const_speed:
    # Resample theta angles so that the velocity is closer to constant.
    lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
    positions = get_positions(theta)

  # Throw away duplicated last position.
  positions = positions[:-1]

  # Set path's up vector to axis closest to average of input pose up vectors.
  avg_up = poses[:, :3, 1].mean(0)
  avg_up = avg_up / np.linalg.norm(avg_up)
  ind_up = np.argmax(np.abs(avg_up))
  up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

  poses = np.stack([viewmatrix(p - center, up, p, lock_up) for p in positions])

  poses = poses @ rotation_about_axis(-render_rotate_yaxis, axis=1)
  poses = poses @ rotation_about_axis(render_rotate_xaxis, axis=0)
  return poses

def pad_poses(p):
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.0], p[Ellipsis, :1, :4].shape)
  return np.concatenate([p[Ellipsis, :3, :4], bottom], axis=-2)


def unpad_poses(p):
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[Ellipsis, :3, :4]


def transform_poses_pca(poses):
  """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
  t = poses[:, :3, 3]
  t_mean = t.mean(axis=0)
  t = t - t_mean

  eigval, eigvec = np.linalg.eig(t.T @ t)
  # Sort eigenvectors in order of largest to smallest eigenvalue.
  inds = np.argsort(eigval)[::-1]
  eigvec = eigvec[:, inds]
  rot = eigvec.T
  if np.linalg.det(rot) < 0:
    rot = np.diag(np.array([1, 1, -1])) @ rot

  transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
  poses_recentered = unpad_poses(transform @ pad_poses(poses))
  transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

  # Flip coordinate system if z component of y-axis is negative
  if poses_recentered.mean(axis=0)[2, 1] < 0:
    poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
    transform = np.diag(np.array([1, -1, -1, 1])) @ transform

  return poses_recentered, transform
