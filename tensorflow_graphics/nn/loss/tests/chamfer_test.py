#Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the chamfer distance loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.nn.loss import chamfer
from tensorflow_graphics.util import test_case


def random_tensor(tensor_shape):
  return np.random.uniform(low=0.0, high=1.0, size=tensor_shape)


def random_tensor_shape():
  tensor_size = np.random.randint(5) + 1
  return np.random.randint(1, 10, size=(tensor_size)).tolist()


def random_point_sets():
  space_dimensions = np.random.randint(10) + 1
  batch_shape = random_tensor_shape()
  point_set_a_size = np.random.randint(100) + 1
  point_set_b_size = np.random.randint(100) + 1

  point_set_a_init = np.random.uniform(
      low=-100.0,
      high=100.0,
      size=batch_shape + [point_set_a_size, space_dimensions])
  point_set_b_init = np.random.uniform(
      low=-100.0,
      high=100.0,
      size=batch_shape + [point_set_b_size, space_dimensions])
  return (point_set_a_init, point_set_b_init)


class ChamferTest(test_case.TestCase):

  @parameterized.parameters(
      ([[0., 0], [0, 1], [1, 0], [-1, 0]],
       [[0., 0], [0, 2], [0.7, 0.4], [-0.5, -0.5]],

       # a[0] -> b[0]
       (0 + \
        # a[1] -> b[2]
        0.7**2 + 0.6**2 + \
        # a[2] -> b[2]
        0.3**2 + 0.4**2 + \
        # a[3] -> b[3]
        0.5) / 4 + \

       # b[0] -> a[0]
       (0 + \
        # b[1] -> a[1]
        1 + \
        # b[2] -> a[2]
        0.3**2 + 0.4**2 + \
        # b[3] -> a[3]
        0.5) / 4),
      ([[0., 1, 4], [3, 4, 2]],
       [[2., 2, 2], [2, 3, 4], [3, 2, 2]],

       # a[0] -> b[1]
       (8 + \
        # a[1] -> b[2]
        4) / 2 + \

       # b[0] -> a[1]
       (5 + \
        # b[1] -> a[1]
        6 + \
        # b[2] -> a[1]
        4) / 3),
  )
  def test_evaluate_preset(self, point_set_a, point_set_b, expected_distance):
    tensor_shape = random_tensor_shape()

    point_set_a = np.tile(point_set_a, tensor_shape + [1, 1])
    point_set_b = np.tile(point_set_b, tensor_shape + [1, 1])
    expected = np.tile(expected_distance, tensor_shape)

    result = chamfer.evaluate(point_set_a, point_set_b)

    self.assertAllClose(expected, result)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_chamfer_evaluate_jacobian_wrt_point_set_a_random(self):
    """Tests the Jacobian of the Chamfer distance loss with respect to the first point set."""
    point_set_a_init, point_set_b_init = random_point_sets()

    point_set_a = tf.convert_to_tensor(value=point_set_a_init)
    point_set_b = tf.convert_to_tensor(value=point_set_b_init)

    self.assert_jacobian_is_correct(
        point_set_a,
        point_set_a_init,
        chamfer.evaluate(point_set_a, point_set_b),
        atol=1e-5)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_chamfer_evaluate_jacobian_wrt_point_set_b_random(self):
    """Tests the Jacobian of the Chamfer distance loss with respect to the second point set."""
    point_set_a_init, point_set_b_init = random_point_sets()

    point_set_a = tf.convert_to_tensor(value=point_set_a_init)
    point_set_b = tf.convert_to_tensor(value=point_set_b_init)

    self.assert_jacobian_is_correct(
        point_set_b,
        point_set_b_init,
        chamfer.evaluate(point_set_a, point_set_b),
        atol=1e-5)

  @parameterized.parameters(
      ("Not all batch dimensions are broadcast-compatible.", (1, 3, 5, 3),
       (2, 4, 3)),
      ("Not all batch dimensions are broadcast-compatible.", (3, 3, 5),
       (2, 4, 5)),
      ("point_set_b must have exactly 3 dimensions in axis -1,.", (2, 4, 3),
       (2, 4, 2)),
      ("point_set_b must have exactly 2 dimensions in axis -1,.", (2, 4, 2),
       (2, 4, 3)),
  )
  def test_evaluate_shape_exception_raised(self, error_msg, *shape):
    """Tests that the shape exception is raised."""
    self.assert_exception_is_raised(chamfer.evaluate, error_msg, shape)

  @parameterized.parameters(
      ((1, 5, 6, 3), (2, 5, 9, 3)),
      ((None, 2, 6, 2), (4, 2, None, 4, 2)),
      ((3, 5, 8, 7), (3, 1, 1, 7)),
  )
  def test_evaluate_shape_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(chamfer.evaluate, shapes)


if __name__ == "__main__":
  test_case.main()
