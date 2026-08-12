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
"""Benchmark target-frame kinematics against whole-model kinematics."""

import time
from typing import Callable
from typing import Sequence

from absl import app
from absl import flags
from etils import epath
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math


_BACKEND = flags.DEFINE_enum(
    'backend', 'frame_pose', ['frame_pose', 'kinematics'], 'kinematics backend'
)
_WORKLOAD = flags.DEFINE_enum(
    'workload', 'jacobian', ['jacobian', 'ik'], 'benchmark workload'
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
      / 'humanoid'
      / 'humanoid.xml'
  )
  model = mujoco.MjModel.from_xml_path(model_path.as_posix())
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  body_id = model.body('hand_right').id
  target_pos = jp.asarray(data.xpos[body_id])
  target_quat = jp.asarray(data.xquat[body_id])
  joint_names = (
      'abdomen_z',
      'abdomen_y',
      'abdomen_x',
      'shoulder1_right',
      'shoulder2_right',
      'elbow_right',
  )
  qpos_indices = jp.asarray(
      [model.joint(name).qposadr.item() for name in joint_names]
  )
  model_mjx = mjx.put_model(model)
  data_mjx = mjx.make_data(model_mjx)

  if _BACKEND.value == 'kinematics':

    def frame_pose(q: jax.Array) -> tuple[jax.Array, jax.Array]:
      qpos = data_mjx.qpos.at[qpos_indices].set(q)
      result = mjx.kinematics(model_mjx, data_mjx.replace(qpos=qpos))
      return result.xpos[body_id], result.xquat[body_id]

  else:

    def frame_pose(q: jax.Array) -> tuple[jax.Array, jax.Array]:
      qpos = data_mjx.qpos.at[qpos_indices].set(q)
      return mjx.frame_pose(
          model_mjx,
          data_mjx.replace(qpos=qpos),
          mujoco.mjtObj.mjOBJ_XBODY,
          body_id,
      )

  def residual(q: jax.Array) -> jax.Array:
    pos, quat = frame_pose(q)
    return jp.concatenate(
        (target_pos - pos, 0.1 * math.quat_sub(target_quat, quat))
    )

  rng = jax.random.key(0)
  qs = 0.2 * jax.random.normal(rng, (_BATCH_SIZE.value, len(joint_names)))

  if _WORKLOAD.value == 'jacobian':

    def pose(q: jax.Array) -> jax.Array:
      pos, quat = frame_pose(q)
      return jp.concatenate((pos, quat))

    evaluate = jax.jit(jax.vmap(jax.jacrev(pose)))
  else:

    def residual_with_aux(q: jax.Array) -> tuple[jax.Array, jax.Array]:
      error = residual(q)
      return error, error

    value_and_jacobian = jax.jacrev(residual_with_aux, has_aux=True)

    def optimize(qs: jax.Array) -> tuple[jax.Array, jax.Array]:
      def iteration(qs: jax.Array, _) -> tuple[jax.Array, None]:
        jacobians, errors = jax.vmap(value_and_jacobian)(qs)
        hessians = jacobians.mT @ jacobians
        rhs = jp.einsum('bij,bi->bj', jacobians, errors)
        identity = jp.eye(qs.shape[-1])
        deltas = jp.linalg.solve(
            hessians + 1e-3 * identity, rhs[..., None]
        ).squeeze(-1)
        proposed = qs - deltas
        proposed_errors = jax.vmap(residual)(proposed)
        current_costs = jp.sum(errors**2, axis=-1)
        proposed_costs = jp.sum(proposed_errors**2, axis=-1)
        accepted = proposed_costs < current_costs
        return jp.where(accepted[:, None], proposed, qs), None

      qs, _ = jax.lax.scan(iteration, qs, None, length=_ITERATIONS.value)
      return qs, jax.vmap(residual)(qs)

    evaluate = jax.jit(optimize)

  lower_time, compile_time, run_time = _measure(evaluate, qs)
  print(f'model: {model_path.name}')
  print('frame: hand_right (XBODY)')
  print(f'workload: {_WORKLOAD.value}')
  print(f'backend: {_BACKEND.value}')
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
