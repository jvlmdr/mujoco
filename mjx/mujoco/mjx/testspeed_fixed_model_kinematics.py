# Copyright 2026 DeepMind Technologies Limited
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
# ==============================================================================
"""Benchmarks differentiated kinematics with a fixed MJX model."""

import time
from typing import Callable
from typing import Sequence

from absl import app
from absl import flags
from etils import epath
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math


_WORKLOAD = flags.DEFINE_enum(
    'workload',
    'jacobian',
    ['forward', 'gradient', 'jacobian', 'ik'],
    'benchmark workload',
)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 2048, 'number of parallel configurations'
)
_ITERATIONS = flags.DEFINE_integer(
    'iterations', 10, 'number of inverse-kinematics iterations'
)


def _measure(
    fn: Callable[[jax.Array], jax.Array | tuple[jax.Array, jax.Array]],
    arg: jax.Array,
) -> tuple[float, float, float]:
  start = time.perf_counter()
  lowered = fn.lower(arg)
  lower_time = time.perf_counter() - start

  start = time.perf_counter()
  compiled = lowered.compile()
  compile_time = time.perf_counter() - start

  run_times = []
  for _ in range(6):
    start = time.perf_counter()
    jax.block_until_ready(compiled(arg))
    run_times.append(time.perf_counter() - start)

  return lower_time, compile_time, min(run_times[1:])


def _main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  jax.config.update('jax_enable_compilation_cache', False)
  model_path = (
      epath.resource_path('mujoco.mjx')
      / 'test_data'
      / 'shadow_hand'
      / 'right_hand.xml'
  )
  model = mujoco.MjModel.from_xml_path(model_path.as_posix())
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  body_id = model.body('rh_thdistal').id
  target_pos = jp.asarray(data.xpos[body_id])
  target_quat = jp.asarray(data.xquat[body_id])
  model_mjx = mjx.put_model(model)
  data_mjx = mjx.make_data(model_mjx)

  def frame_pose(qpos: jax.Array) -> tuple[jax.Array, jax.Array]:
    result = mjx.kinematics(model_mjx, data_mjx.replace(qpos=qpos))
    return result.xpos[body_id], result.xquat[body_id]

  def residual(qpos: jax.Array) -> jax.Array:
    pos, quat = frame_pose(qpos)
    return jp.concatenate(
        (target_pos - pos, 0.1 * math.quat_sub(target_quat, quat))
    )

  rng = jax.random.key(0)
  qpos = 0.2 * jax.random.normal(rng, (_BATCH_SIZE.value, model.nq))

  if _WORKLOAD.value == 'forward':
    evaluate = jax.jit(jax.vmap(frame_pose))
  elif _WORKLOAD.value == 'gradient':

    def loss(qpos: jax.Array) -> jax.Array:
      error = residual(qpos)
      return 0.5 * jp.vdot(error, error)

    evaluate = jax.jit(jax.vmap(jax.value_and_grad(loss)))
  elif _WORKLOAD.value == 'jacobian':

    def pose(qpos: jax.Array) -> jax.Array:
      pos, quat = frame_pose(qpos)
      return jp.concatenate((pos, quat))

    evaluate = jax.jit(jax.vmap(jax.jacrev(pose)))
  else:

    def residual_with_aux(qpos: jax.Array) -> tuple[jax.Array, jax.Array]:
      error = residual(qpos)
      return error, error

    value_and_jacobian = jax.jacrev(residual_with_aux, has_aux=True)

    def optimize(qpos: jax.Array) -> tuple[jax.Array, jax.Array]:
      def iteration(qpos: jax.Array, _) -> tuple[jax.Array, None]:
        jacobians, errors = jax.vmap(value_and_jacobian)(qpos)
        hessians = jacobians.mT @ jacobians
        rhs = jp.einsum('bij,bi->bj', jacobians, errors)
        identity = jp.eye(qpos.shape[-1])
        deltas = jp.linalg.solve(
            hessians + 1e-3 * identity, rhs[..., None]
        ).squeeze(-1)
        proposed = qpos - deltas
        proposed_errors = jax.vmap(residual)(proposed)
        current_costs = jp.sum(errors**2, axis=-1)
        proposed_costs = jp.sum(proposed_errors**2, axis=-1)
        accepted = proposed_costs < current_costs
        return jp.where(accepted[:, None], proposed, qpos), None

      qpos, _ = jax.lax.scan(iteration, qpos, None, length=_ITERATIONS.value)
      return qpos, jax.vmap(residual)(qpos)

    evaluate = jax.jit(optimize)

  lower_time, compile_time, run_time = _measure(evaluate, qpos)
  print(f'model: {model_path.name}')
  print('frame: rh_thdistal (XBODY)')
  print(f'workload: {_WORKLOAD.value}')
  print(f'batch size: {_BATCH_SIZE.value}')
  if _WORKLOAD.value == 'ik':
    print(f'iterations: {_ITERATIONS.value}')
  print(f'lowering: {lower_time:.3f} s')
  print(f'compilation: {compile_time:.3f} s')
  print(f'warm execution: {1e3 * run_time:.3f} ms')


def main():
  app.run(_main)


if __name__ == '__main__':
  main()
