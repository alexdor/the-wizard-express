from collections import namedtuple

from tensorflow import reshape, shape
from tensorflow.compat import dimension_value


def get_shape(tensor, dim=None):
    """Gets the most specific dimension size(s) of the given tensor.

    This is a wrapper around the two ways to get the shape of a tensor: (1)
    t.get_shape() to get the static shape, and (2) tf.shape(t) to get the dynamic
    shape. This function returns the most specific available size for each
    dimension. If the static size is available, that is returned. Otherwise, the
    tensor representing the dynamic size is returned.

    Args:
      tensor: Input tensor.
      dim: Desired dimension. Use None to retrieve the list of all sizes.

    Returns:
      output = Most specific static dimension size(s).
    """
    static_shape = tensor.get_shape()
    dynamic_shape = shape(tensor)
    if dim is not None:
        return dimension_value(static_shape[dim]) or dynamic_shape[dim]

    return [dimension_value(d) or dynamic_shape[i] for i, d in enumerate(static_shape)]


BertInputs = namedtuple("BertInputs", ["token_ids", "mask", "segment_ids"])


def tensor_flatten(t):
    """Collapse all dimensions of a tensor but the last.

    This function also returns a function to unflatten any tensors to recover
    the original shape (aside from the last dimension). This is useful for
    interfacing with functions that are agnostic to all dimensions but the last.
    For example, if we want to apply a linear projection to a batched sequence
    of embeddings:

      t = tf.random_uniform([batch_size, sequence_length, embedding_size])
      w = tf.get_variable("w", [embedding_size, projection_size])
      flat_t, unflatten = flatten(t)
      flat_projected = tf.matmul(flat_t, w)
      projected = unflatten(flat_projected)

    Args:
      t: [dim_1, ..., dim_(n-1), dim_n]

    Returns:
      output: [dim_1 * ... * dim_(n-1), dim_n]
      _unflatten: A function that when called with a flattened tensor returns the
          unflattened version, i.e. reshapes any tensor with shape
          [dim_1 * ... * dim_(n-1), dim_new] to [dim_1, ..., dim_(n-1), dim_new].
    """
    input_shape = get_shape(t)
    if len(input_shape) > 2:
        t = reshape(t, [-1, input_shape[-1]])

    def _unflatten(flat_t):
        if len(input_shape) > 2:
            return reshape(flat_t, input_shape[:-1] + [get_shape(flat_t, -1)])
        return flat_t

    return t, _unflatten


def flatten_bert_inputs(bert_inputs):
    """Flatten all tensors in a BertInput and also return the inverse."""
    flat_token_ids, unflatten = tensor_flatten(bert_inputs.token_ids)
    flat_mask, _ = tensor_flatten(bert_inputs.mask)
    flat_segment_ids, _ = tensor_flatten(bert_inputs.segment_ids)
    flat_bert_inputs = BertInputs(
        token_ids=flat_token_ids, mask=flat_mask, segment_ids=flat_segment_ids
    )
    return flat_bert_inputs, unflatten

    # def _read(self, joint_inputs):
    #     # [batch_size * num_candidates, query_seq_len + candidate_seq_len]
    #     flat_joint_inputs, _ = flatten_bert_inputs(joint_inputs)
    #     # [batch_size * num_candidates, num_masks]
    #     flat_mlm_positions, _ = tensor_flatten(
    #         tf.tile(tf.expand_dims(mlm_positions, 1), [1, params["num_candidates"], 1])
    #     )

    #     batch_size, num_masks = tensor_utils.shape(mlm_targets)

    #     # [batch_size * num_candidates, query_seq_len + candidates_seq_len]
    #     flat_joint_bert_outputs = bert_module(
    #         inputs=dict(
    #             input_ids=flat_joint_inputs.token_ids,
    #             input_mask=flat_joint_inputs.mask,
    #             segment_ids=flat_joint_inputs.segment_ids,
    #             mlm_positions=flat_mlm_positions,
    #         ),
    #         signature="mlm",
    #         as_dict=True,
    #     )

    #     # [batch_size, num_candidates]
    #     candidate_score = retrieval_score

    #     # [batch_size, num_candidates]
    #     candidate_log_probs = tf.math.log_softmax(candidate_score)
