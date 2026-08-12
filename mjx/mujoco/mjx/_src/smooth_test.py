# Copyright 2023 DeepMind Technologies Limited
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
"""Tests for smooth dynamics functions."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import test_util
from mujoco.mjx._src.types import ConeType  # pylint: disable=g-importing-member
from mujoco.mjx._src.types import JacobianType  # pylint: disable=g-importing-member
import numpy as np

# tolerance for difference between MuJoCo and MJX smooth calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f'mismatch: {name}'
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


def _assert_attr_eq(a, b, attr):
  _assert_eq(getattr(a, attr), getattr(b, attr), attr)


class SmoothTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # although we already have generous padding of thresholds, it doesn't hurt
    # to also fix the seed to reduce test flakiness
    np.random.seed(0)

  def test_smooth(self):
    """Tests MJX smooth functions match MuJoCo smooth functions."""

    m = test_util.load_test_file('pendula.xml')
    # tell MJX to use sparse mass matrices:
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    d = mujoco.MjData(m)
    # give the system a little kick to ensure we have non-identity rotations
    d.qvel = np.random.random(m.nv)
    mujoco.mj_step(m, d, 10)  # let dynamics get state significantly non-zero
    # randomize mocap
    d.mocap_pos = np.random.random(d.mocap_pos.shape)
    d.mocap_quat = np.random.random(d.mocap_quat.shape)
    mujoco.mj_forward(m, d)
    mx = mjx.put_model(m)

    # kinematics
    dx = jax.jit(mjx.kinematics)(mx, mjx.put_data(m, d))
    _assert_attr_eq(d, dx, 'xanchor')
    _assert_attr_eq(d, dx, 'xaxis')
    _assert_attr_eq(d, dx, 'xpos')
    _assert_attr_eq(d, dx, 'xquat')
    _assert_eq(d.xmat.reshape((-1, 3, 3)), dx.xmat, 'xmat')
    _assert_attr_eq(d, dx, 'xipos')
    _assert_eq(d.ximat.reshape((-1, 3, 3)), dx.ximat, 'ximat')
    _assert_attr_eq(d, dx, 'geom_xpos')
    _assert_eq(d.geom_xmat.reshape((-1, 3, 3)), dx.geom_xmat, 'geom_xmat')
    _assert_attr_eq(d, dx, 'site_xpos')
    _assert_eq(d.site_xmat.reshape((-1, 3, 3)), dx.site_xmat, 'site_xmat')
    # com_pos
    dx = jax.jit(mjx.com_pos)(mx, mjx.put_data(m, d))
    _assert_attr_eq(d, dx, 'subtree_com')
    _assert_attr_eq(d, dx._impl, 'cinert')
    _assert_attr_eq(d, dx._impl, 'cdof')
    # camlight
    dx = jax.jit(mjx.camlight)(mx, mjx.put_data(m, d))
    _assert_attr_eq(d, dx, 'cam_xpos')
    _assert_eq(d.cam_xmat.reshape((-1, 3, 3)), dx.cam_xmat, 'cam_xmat')
    # crb
    dx = jax.jit(mjx.crb)(mx, mjx.put_data(m, d))
    _assert_attr_eq(d, dx._impl, 'crb')
    _assert_attr_eq(d, dx._impl, 'qM')
    # factor_m
    dx = jax.jit(mjx.factor_m)(mx, mjx.put_data(m, d))
    qLDLegacy = np.zeros(mx.nM)  # pylint:disable=invalid-name
    for i in range(m.nC):
      qLDLegacy[m.mapM2M[i]] = d.qLD[i]
    _assert_eq(qLDLegacy, dx._impl.qLD, 'qLD')
    _assert_attr_eq(d, dx._impl, 'qLDiagInv')
    # com_vel
    dx = jax.jit(mjx.com_vel)(mx, mjx.put_data(m, d))
    _assert_attr_eq(d, dx, 'cvel')
    _assert_attr_eq(d, dx._impl, 'cdof_dot')
    # rne
    dx = jax.jit(mjx.rne)(mx, mjx.put_data(m, d))
    _assert_attr_eq(d, dx, 'qfrc_bias')
    # rne (flg_acc=True)
    qfrc_bias = np.zeros(m.nv)
    mujoco.mj_rne(m, d, 1, qfrc_bias)
    dx = jax.jit(mjx.rne, static_argnums=(2,))(
        mx, mjx.put_data(m, d), flg_acc=True
    )
    _assert_eq(dx.qfrc_bias, qfrc_bias, 'qfrc_bias')

    # set dense jacobian for tendon:
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    d = mujoco.MjData(m)
    # give the system a little kick to ensure we have non-identity rotations
    d.qvel = np.random.random(m.nv)
    mujoco.mj_step(m, d, 10)  # let dynamics get state significantly non-zero
    mujoco.mj_forward(m, d)
    # tendon
    dx = jax.jit(mjx.tendon)(mx, mjx.put_data(m, d))
    _assert_attr_eq(d, dx._impl, 'ten_J')
    _assert_attr_eq(d, dx, 'ten_length')
    # transmission
    dx = jax.jit(mjx.transmission)(mx, dx)
    _assert_attr_eq(d, dx._impl, 'actuator_length')

    # convert sparse actuator_moment to dense representation
    moment = np.zeros((m.nu, m.nv))
    mujoco.mju_sparse2dense(
        moment,
        d.actuator_moment,
        d.moment_rownnz,
        d.moment_rowadr,
        d.moment_colind,
    )
    _assert_eq(moment, dx._impl.actuator_moment, 'actuator_moment')

  def test_fixed_scalar_tree_kinematics(self):
    model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <worldbody>
            <body pos="0.2 -0.1 0.3" euler="0.1 -0.2 0.3">
              <geom type="sphere" size="0.03"/>
              <body pos="0.1 0.2 -0.1" euler="-0.2 0.1 0.15">
                <joint type="hinge" ref="0.2" pos="0.03 -0.02 0.01"
                       axis="0.2 0.7 -0.4"/>
                <geom type="sphere" size="0.02"/>
                <body pos="0.05 -0.1 0.08" euler="0.2 0.1 -0.1">
                  <site pos="0.08 0.03 -0.02" euler="0.1 0.2 0.3"/>
                  <body pos="0.04 0.02 0.07" euler="-0.1 0.2 0.1">
                    <joint type="slide" ref="-0.1" axis="0.4 -0.3 0.8"/>
                    <geom type="capsule" size="0.02 0.05"/>
                    <site pos="0.03 -0.02 0.09"/>
                  </body>
                  <body pos="-0.02 0.06 0.04" euler="0.3 -0.1 0.2">
                    <joint type="hinge" axis="-0.6 0.1 0.5"/>
                    <geom type="cylinder" size="0.02 0.04"/>
                    <site pos="0.01 0.04 0.05"/>
                  </body>
                </body>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """)
    model_mjx = mjx.put_model(model)
    data_mjx = mjx.make_data(model_mjx)
    qpos = jp.array((0.73, -0.19, -0.44))

    def outputs(data):
      return (
          data.xanchor,
          data.xaxis,
          data.xpos,
          data.xquat,
          data.xmat,
          data.xipos,
          data.ximat,
          data.geom_xpos,
          data.geom_xmat,
          data.site_xpos,
          data.site_xmat,
      )

    def fixed_model(qpos):
      return outputs(mjx.kinematics(model_mjx, data_mjx.replace(qpos=qpos)))

    def dynamic_model(model_mjx, qpos):
      return outputs(mjx.kinematics(model_mjx, data_mjx.replace(qpos=qpos)))

    actual = jax.jit(fixed_model)(qpos)
    expected = jax.jit(dynamic_model)(model_mjx, qpos)
    jax.tree_util.tree_map(
        lambda actual, expected: _assert_eq(actual, expected, 'kinematics'),
        actual,
        expected,
    )

    def pose_vector(outputs):
      xpos, xquat, site_xpos, site_xmat = (
          outputs[2],
          outputs[3],
          outputs[9],
          outputs[10],
      )
      return jp.concatenate(
          tuple(
              value.reshape(-1) for value in (xpos, xquat, site_xpos, site_xmat)
          )
      )

    fixed_poses = lambda qpos: pose_vector(fixed_model(qpos))
    dynamic_poses = lambda model, qpos: pose_vector(dynamic_model(model, qpos))
    dynamic_poses = jax.jit(dynamic_poses)
    expected_jacobian = jax.jacrev(dynamic_poses, argnums=1)(model_mjx, qpos)
    reverse_jacobian = jax.jacrev(jax.jit(fixed_poses))(qpos)
    forward_jacobian = jax.jacfwd(jax.jit(fixed_poses))(qpos)
    _assert_eq(reverse_jacobian, expected_jacobian, 'reverse Jacobian')
    _assert_eq(forward_jacobian, expected_jacobian, 'forward Jacobian')

  def test_disable_gravity(self):
    m = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <option>
            <flag gravity="disable"/>
          </option>
          <worldbody>
            <body>
              <joint type="free"/>
              <geom size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    dx = jax.jit(mjx.rne)(mx, dx)
    np.testing.assert_allclose(dx.qfrc_bias, 0)

  def test_site_transmission(self):
    m = mujoco.MjModel.from_xml_string("""
        <mujoco>
        <compiler autolimits="true"/>
        <worldbody>
          <body>
            <joint type="free"/>
            <geom type="box" size=".05 .05 .05" mass="1"/>
            <site name="site1"/>
            <body>
              <joint type="hinge"/>
              <geom size="0.1" mass="1"/>
              <site name="site2" pos="0.1 0.2 0.3"/>
            </body>
          </body>
          <body pos="1 0 0">
            <joint name="slide" type="hinge"/>
            <geom type="box" size=".05 .05 .05" mass="1"/>
          </body>
        </worldbody>
        <actuator>
          <position site="site1" kv="0.1" gear="1 2 3 0 0 0"/>
          <position site="site1" kv="0.2" gear="0 0 0 1 2 3"/>
          <position site="site2" kv="0.3" gear="0 3 0 0 0 1"/>
          <position joint="slide" kv="0.05" />
          <position site="site2" refsite="site1" gear="1 2 3 0.5 0.4 0.6"/>
        </actuator>
        </mujoco>
      """)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    mujoco.mj_transmission(m, d)
    dx = jax.jit(mjx.transmission)(mx, dx)
    _assert_attr_eq(d, dx._impl, 'actuator_length')

    # convert sparse actuator_moment to dense representation
    moment = np.zeros((m.nu, m.nv))
    mujoco.mju_sparse2dense(
        moment,
        d.actuator_moment,
        d.moment_rownnz,
        d.moment_rowadr,
        d.moment_colind,
    )
    _assert_eq(moment, dx._impl.actuator_moment, 'actuator_moment')

  def test_subtree_vel(self):
    """Tests MJX subtree_vel function matches MuJoCo mj_subtreeVel."""

    m = test_util.load_test_file('humanoid/humanoid.xml')
    d = mujoco.MjData(m)
    # give the system a little kick to ensure we have non-identity rotations
    d.qvel = np.random.random(m.nv)
    mujoco.mj_step(m, d, 10)  # let dynamics get state significantly non-zero
    mujoco.mj_forward(m, d)
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    # subtree velocity
    mujoco.mj_subtreeVel(m, d)
    dx = jax.jit(mjx.subtree_vel)(mx, dx)

    _assert_attr_eq(d, dx._impl, 'subtree_linvel')
    _assert_attr_eq(d, dx._impl, 'subtree_angmom')


class RnePostConstraintTest(parameterized.TestCase):
  _CONNECT_SITE = """
    <equality>
      <connect site1="site1" site2="site2"/>
    </equality>
    """
  _CONNECT_BODY = """
    <equality>
      <connect body1="body1" body2="body2" anchor="1 2 3"/>
    </equality>
    """
  _WELD_SITE = """
    <equality>
      <weld site1="site1" site2="site2"/>
    </equality>
    """
  _WELD_BODY = """
    <equality>
      <weld body1="body1" body2="body2"/>
    </equality>
    """
  _CONNECT_SITE_WELD_SITE = """
    <equality>
      <connect site1="site1" site2="site2"/>
      <weld site1="site1" site2="site2"/>
    </equality>
    """
  _WELD_SITE_CONNECT_SITE = """
    <equality>
      <weld site1="site1" site2="site2"/>
      <connect site1="site1" site2="site2"/>
    </equality>
    """
  _WELD_SITE_CONNECT_SITE_WELD_BODY = """
    <equality>
      <weld site1="site1" site2="site2"/>
      <connect site1="site1" site2="site2"/>
      <weld body1="body1" body2="body2"/>
    </equality>
    """
  _CONNECT_SITE_WELD_SITE_WELD_BODY = """
    <equality>
      <connect site1="site1" site2="site2"/>
      <weld site1="site1" site2="site2"/>
      <weld body1="body1" body2="body2"/>
    </equality>
    """
  _CONNECT_SITE_CONNECT_BODY_CONNECT_WELD = """
    <equality>
      <connect site1="site1" site2="site2"/>
      <connect body1="body1" body2="body2" anchor="1 2 3"/>
      <weld body1="body1" body2="body2"/>
    </equality>
    """

  @parameterized.parameters(
      ('', ConeType.PYRAMIDAL, None),
      ('', ConeType.ELLIPTIC, None),
      (_CONNECT_SITE, ConeType.PYRAMIDAL, None),
      (_CONNECT_BODY, ConeType.PYRAMIDAL, None),
      (_WELD_SITE, ConeType.PYRAMIDAL, None),
      (_WELD_BODY, ConeType.PYRAMIDAL, None),
      (_CONNECT_SITE_WELD_SITE, ConeType.PYRAMIDAL, None),
      (
          _WELD_SITE_CONNECT_SITE,
          ConeType.PYRAMIDAL,
          np.array([6, 7, 8, 0, 1, 2, 3, 4, 5]),
      ),
      (
          _WELD_SITE_CONNECT_SITE_WELD_BODY,
          ConeType.PYRAMIDAL,
          np.array([6, 7, 8, 0, 1, 2, 3, 4, 5]),
      ),
      (_CONNECT_SITE_WELD_SITE_WELD_BODY, ConeType.PYRAMIDAL, None),
      (_CONNECT_SITE_CONNECT_BODY_CONNECT_WELD, ConeType.PYRAMIDAL, None),
  )
  def test_rnepostconstraint(self, equality, cone_type, efc_map):
    """Tests MJX rne_postconstraint function to match MuJoCo mj_rnePostConstraint."""

    m = mujoco.MjModel.from_xml_string(f"""
        <mujoco>
          <worldbody>
            <geom name="floor" size="10 10 .05" type="plane"/>
            <site name="site1"/>
            <body name="body1">
            </body>
            <body pos="0 0 1" name="body2">
              <joint type="ball" damping="1"/>
              <geom type="capsule" size="0.1 0.5" fromto="0 0 0 0.5 0 0" condim="1"/>
              <body pos="0.5 0 0">
                <joint type="ball" damping="1"/>
                <geom type="capsule" size="0.1 0.5" fromto="0 0 0 0.5 0 0"  condim="3"/>
                <site name="site2"/>
              </body>
            </body>
            <body pos="0 1 1">
              <joint type="ball" damping="1"/>
              <geom type="capsule" size="0.1 0.5" fromto="0 0 0 0.5 0 0" condim="6"/>
              <body pos="0.5 0 0">
                <joint type="ball" damping="1"/>
                <geom type="capsule" size="0.1 0.5" fromto="0 0 0 0.5 0 0"  condim="3"/>
              </body>
            </body>
          </worldbody>
          {equality}
          <keyframe>
            <key qpos='0.424577 0.450592 0.451703 -0.642391 0.729379 0.545151 0.407756 0.0674697 0.424577 1.450592 0.451703 -0.642391 0.729379 0.545151 0.407756 0.0674697'/>
          </keyframe>
        </mujoco>
    """)
    # set cone type
    m.opt.cone = cone_type
    # create data and set to keyframe
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    # apply external forces
    d.xfrc_applied = 0.001 * np.ones(d.xfrc_applied.shape)
    mujoco.mj_step(m, d, 2)
    mujoco.mj_forward(m, d)
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    if efc_map is not None:
      efc_force = d.efc_force.copy()
      efc_force[: len(efc_map)] = d.efc_force[efc_map]
      dx = dx.tree_replace({'_impl.efc_force': jp.array(efc_force)})

    # rne postconstraint
    mujoco.mj_rnePostConstraint(m, d)
    dx = jax.jit(mjx.rne_postconstraint)(mx, dx)

    _assert_eq(d.cacc, dx._impl.cacc, 'cacc')
    _assert_eq(d.cfrc_ext, dx._impl.cfrc_ext, 'cfrc_ext')
    _assert_eq(d.cfrc_int, dx._impl.cfrc_int, 'cfrc_int')


class TendonTest(parameterized.TestCase):

  @parameterized.parameters(
      'tendon/fixed.xml',
      'tendon/fixed_site.xml',
      'tendon/fixed_site_wrap.xml',
      'tendon/pulley_fixed_site_wrap.xml',
      'tendon/pulley_site.xml',
      'tendon/pulley_site_wrap.xml',
      'tendon/pulley_wrap.xml',
      'tendon/no_tendon.xml',
      'tendon/site.xml',
      'tendon/site_wrap.xml',
      'tendon/tendon.xml',
      'tendon/wrap_sidesite.xml',
  )
  def test_tendon(self, filename):
    """Tests MJX tendon function matches MuJoCo mj_tendon."""
    m = test_util.load_test_file(filename)
    d = mujoco.MjData(m)
    # give the system a little kick to ensure we have non-identity rotations
    d.qvel = np.random.random(m.nv)
    mujoco.mj_step(m, d, 10)  # let dynamics get state significantly non-zero
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    mujoco.mj_forward(m, d)
    dx = jax.jit(mjx.forward)(mx, dx)

    _assert_eq(d.ten_length, dx.ten_length, 'ten_length')
    _assert_eq(d.ten_J, dx._impl.ten_J, 'ten_J')
    _assert_eq(d.ten_wrapnum, dx._impl.ten_wrapnum, 'ten_wrapnum')
    _assert_eq(d.ten_wrapadr, dx._impl.ten_wrapadr, 'ten_wrapadr')
    _assert_eq(d.wrap_obj, dx._impl.wrap_obj, 'wrap_obj')
    _assert_eq(d.wrap_xpos, dx._impl.wrap_xpos, 'wrap_xpos')

  @parameterized.parameters(JacobianType.DENSE, JacobianType.SPARSE)
  def test_tendon_armature(self, jacobian):
    """Tests MJX tendon armature matches MuJoCo."""
    m = test_util.load_test_file('tendon/armature.xml')
    m.opt.jacobian = jacobian
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    dx = dx.tree_replace(
        {'_impl.qM': jp.zeros((m.nv, m.nv)), 'qfrc_bias': jp.zeros(m.nv)}
    )

    dx = mjx.crb(mx, dx)
    dx = mjx.tendon_armature(mx, dx)

    if jacobian == JacobianType.DENSE:
      qM = np.zeros((m.nv, m.nv))  # pylint: disable=invalid-name
      mujoco.mj_fullM(m, qM, d.qM)
    else:
      qM = d.qM  # pylint: disable=invalid-name
    _assert_eq(dx._impl.qM, qM, 'qM')

    dx = mjx.rne(mx, dx)
    dx = mjx.tendon_bias(mx, dx)
    _assert_eq(dx.qfrc_bias, d.qfrc_bias, 'qfrc_bias')


if __name__ == '__main__':
  absltest.main()
