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
"""This module implements the fscore metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def evaluate(ground_truth_labels, predicted_labels, name=None):
  """Computes the fscore metric for the given ground truth and predicted labels.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Args:
    ground_truth_labels: A tensor of shape `[A1, ..., An]`. Each attribute can
      either be 0 or 1.
    predicted_labels: A tensor of shape `[A1, ..., An]`. Each attribute can
      either be 0 or 1.
    name: A name for this op. Defaults to "fscore_evaluate".

  Returns:
    A tensor of shape `[A1, ..., An]` that stores the fscore metric for the
    given ground truth labels and predictions.

  Raises:
    ValueError: if the shape of `ground_truth_labels`, `predicted_labels` is
    not supported.
  """
  with tf.compat.v1.name_scope(name, "fscore_evaluate",
                               [ground_truth_labels, predicted_labels]):
    ground_truth_labels = tf.convert_to_tensor(value=ground_truth_labels)
    predicted_labels = tf.convert_to_tensor(value=predicted_labels)

    shape.compare_batch_dimensions(
        tensors=(ground_truth_labels, predicted_labels),
        tensor_names=("ground_truth_labels", "predicted_labels"),
        last_axes=-1,
        broadcast_compatible=True)

    ground_truth_labels = asserts.assert_binary(ground_truth_labels)
    predicted_labels = asserts.assert_binary(predicted_labels)

    sum_ground_truth = tf.math.reduce_sum(
        input_tensor=ground_truth_labels, axis=-1)
    # Verify that the ground truth labels are not all zeros.
    sum_ground_truth = asserts.assert_all_above(
        sum_ground_truth, 0, open_bound=True)
    true_positives = tf.math.reduce_sum(
        input_tensor=ground_truth_labels * predicted_labels, axis=-1)
    total_positives = tf.math.reduce_sum(input_tensor=predicted_labels, axis=-1)

    recall = true_positives / sum_ground_truth
    precision = tf.compat.v1.where(
        tf.math.equal(total_positives, 0), tf.zeros_like(recall),
        true_positives / total_positives)

    return tf.compat.v1.where(
        tf.math.equal(true_positives, 0), tf.zeros_like(recall),
        2 * (precision * recall) / (precision + recall))


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
