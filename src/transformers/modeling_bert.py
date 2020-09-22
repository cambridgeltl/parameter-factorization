"""PyTorch BERT model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import os
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from .modeling_utils import (PretrainedConfig, PreTrainedModel,
                             prune_linear_layer, add_start_docstrings)

logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
}

BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-config.json",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-config.json",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-config.json",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-config.json",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-config.json",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-config.json",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-config.json",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-config.json",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-config.json",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-config.json",
}


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
                     "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                la = re.split(r'_(\d+)', m_name)
            else:
                la = [m_name]
            if la[0] == 'kernel' or la[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif la[0] == 'output_bias' or la[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif la[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif la[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, la[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(la) >= 2:
                num = int(la[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(PretrainedConfig):
    r"""
        :class:`~pytorch_transformers.BertConfig` is the configuration class to store the configuration of a
        `BertModel`.


        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """
    pretrained_config_archive_map = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 **kwargs):
        super(BertConfig, self).__init__(**kwargs)
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                                                               and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")

    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # Update hyper params
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # outputs, (hidden states), (attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def __init__(self, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__(*inputs, **kwargs)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


BERT_START_DOCSTRING = r"""    The BERT model was proposed in
    `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_
    by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. It's a bidirectional transformer
    pre-trained using a combination of masked language modeling objective and next sentence prediction
    on a large corpus comprising the Toronto Book Corpus and Wikipedia.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`:
        https://arxiv.org/abs/1810.04805

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~pytorch_transformers.BertConfig`): Model configuration class with all the parameters of the model.
"""

BERT_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, BERT input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]``

                ``token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``

                ``token_type_ids:   0   0   0   0  0     0   0``

            Indices can be obtained using :class:`pytorch_transformers.BertTokenizer`.
            See :func:`pytorch_transformers.PreTrainedTokenizer.encode` and
            :func:`pytorch_transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""


@add_start_docstrings("The bare Bert Model transformer outputing raw hidden-states without any specific head on top.",
                      BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertModel(BertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.apply(self.init_weights)

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers
        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, do=0.):
        super(MLP, self).__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(do)

    def forward(self, x):
        out = self.dense(x)
        out = self.relu(out)
        out = self.dropout(out)

        return out


class SVIMetaNet(nn.Module):
    def __init__(self, n_lang, n_classes, hid_dim, emb_dim, dropout,
                 lang_prior_sigma=1., task_prior_sigma=1., theta_prior_sigma=1.,
                 n_layers=6):
        super().__init__()
        self.n_lang = n_lang
        self.n_classes = n_classes
        self.lang_prior_sigma = lang_prior_sigma
        self.task_prior_sigma = task_prior_sigma
        self.theta_prior_sigma = theta_prior_sigma
        assert n_layers >= 2
        self.n_layers = n_layers

        self.lang_mean = nn.Embedding(n_lang, emb_dim)
        self.lang_logvar = nn.Embedding(n_lang, emb_dim)
        self.task_mean = nn.Embedding(len(n_classes), emb_dim)
        self.task_logvar = nn.Embedding(len(n_classes), emb_dim)
        self.MLP_in = MLP(emb_dim * 4, hid_dim, do=dropout)
        for lyr in range(2, n_layers):
            setattr(self, "MLP_{}".format(lyr), MLP(hid_dim, hid_dim, do=dropout))
        self.MLP_out = MLP(hid_dim, emb_dim, do=dropout)
        for task, n_cl in n_classes.items():
            setattr(self, "emb2weight_mean_{}".format(task), nn.Linear(emb_dim, n_cl * hid_dim))
            setattr(self, "emb2weight_logvar_{}".format(task), nn.Linear(emb_dim, n_cl * hid_dim))
            setattr(self, "emb2bias_{}".format(task), nn.Linear(emb_dim, n_cl))
        self.normal = torch.distributions.Normal(0, 1)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.lang_mean.weight.data, mean=0.0, std=0.1)
        torch.nn.init.normal_(self.task_mean.weight.data, mean=0.0, std=0.1)
        torch.nn.init.uniform_(self.lang_logvar.weight.data, a=0.0, b=0.5)
        torch.nn.init.uniform_(self.task_logvar.weight.data, a=0.0, b=0.5)

    def softrelu(self, x):
        return torch.log1p(torch.exp(x))

    def kl_div(self, lv, mu, sigma):
        prior_sigma = getattr(self, "{}_prior_sigma".format(lv))
        kl = math.log(prior_sigma) - torch.log(sigma).sum() - mu.numel() + ((sigma + mu**2) / prior_sigma).sum()
        return 0.5 * kl

    def sample_weight(self, lv, mu, sigma, sample, calculate_log_probs):
        if sample:
            epsilon = self.normal.sample(sigma.shape)
            epsilon = epsilon.cuda() if mu.is_cuda else epsilon
            weight = mu + sigma * epsilon
        else:
            weight = mu

        kl = self.kl_div(lv, mu, sigma) if calculate_log_probs else 0.
        return weight, kl

    def forward(self, input, task, task_idx, language, sample, calculate_log_probs):
        lang_mean = self.lang_mean(language)
        lang_var = self.softrelu(self.lang_logvar(language))
        task_mean = self.task_mean(task_idx)
        task_var = self.softrelu(self.task_logvar(task_idx))
        lang_emb, kl_l = self.sample_weight("lang", lang_mean, lang_var, sample, calculate_log_probs)
        task_emb, kl_t = self.sample_weight("task", task_mean, task_var, sample, calculate_log_probs)
        emb = torch.cat((lang_emb, task_emb, lang_emb - task_emb, lang_emb * task_emb), -1)
        emb = self.MLP_in(emb)
        for lyr in range(2, self.n_layers):
            MLP = getattr(self, "MLP_{}".format(lyr))
            emb = MLP(emb)
        emb = self.MLP_out(emb)
        emb2weight_mean = getattr(self, "emb2weight_mean_{}".format(task))
        emb2weight_logvar = getattr(self, "emb2weight_logvar_{}".format(task))
        emb2bias = getattr(self, "emb2bias_{}".format(task))
        weight_mean = emb2weight_mean(emb)
        weight_var = self.softrelu(emb2weight_logvar(emb))
        weight, kl_w = self.sample_weight("theta", weight_mean, weight_var, sample, False)
        weight = weight.view(self.n_classes[task], -1)
        bias = emb2bias(emb).view(self.n_classes[task]).squeeze()
        logits = F.linear(input, weight, bias)
        kl = kl_l + kl_t + kl_w
        return logits, kl


class LRCMetaNet(nn.Module):
    def __init__(self, n_lang, n_classes, hid_dim, emb_dim, dropout,
                 lang_prior_sigma=1., task_prior_sigma=1., theta_prior_sigma=1.,
                 n_layers=6, rank_cov=0):
        super().__init__()
        self.n_lang = n_lang
        self.n_classes = n_classes
        self.emb_dim = emb_dim
        self.lang_prior_sigma = lang_prior_sigma
        self.task_prior_sigma = task_prior_sigma
        self.theta_prior_sigma = theta_prior_sigma
        assert n_layers >= 2
        assert rank_cov >= 1
        self.n_layers = n_layers
        self.rank_cov = rank_cov

        self.lang_mean = nn.Parameter(torch.Tensor(n_lang * emb_dim))
        self.lang_logvar = nn.Parameter(torch.Tensor(n_lang * emb_dim))
        self.lang_fctvar = nn.Parameter(torch.Tensor(n_lang * emb_dim, rank_cov))
        self.task_mean = nn.Parameter(torch.Tensor(len(n_classes) * emb_dim))
        self.task_logvar = nn.Parameter(torch.Tensor(len(n_classes) * emb_dim))
        self.task_fctvar = nn.Parameter(torch.Tensor(len(n_classes) * emb_dim, rank_cov))

        self.MLP_in = MLP(emb_dim * 4, hid_dim, do=dropout)
        for lyr in range(2, n_layers):
            setattr(self, "MLP_{}".format(lyr), MLP(hid_dim, hid_dim, do=dropout))
        self.MLP_out = MLP(hid_dim, emb_dim, do=dropout)
        for task, n_cl in n_classes.items():
            setattr(self, "emb2weight_mean_{}".format(task), nn.Linear(emb_dim, n_cl * hid_dim))
            setattr(self, "emb2weight_logvar_{}".format(task), nn.Linear(emb_dim, n_cl * hid_dim))
            setattr(self, "emb2bias_{}".format(task), nn.Linear(emb_dim, n_cl))
        self.normal = torch.distributions.Normal(0, 1)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.lang_mean.data, mean=0.0, std=0.1)
        torch.nn.init.normal_(self.task_mean.data, mean=0.0, std=0.1)
        torch.nn.init.uniform_(self.lang_logvar.data, a=0.0, b=0.5)
        torch.nn.init.uniform_(self.task_logvar.data, a=0.0, b=0.5)
        torch.nn.init.uniform_(self.lang_fctvar.data, a=0.0, b=math.sqrt(0.5))
        torch.nn.init.uniform_(self.task_fctvar.data, a=0.0, b=math.sqrt(0.5))

    def softrelu(self, x):
        return torch.log1p(torch.exp(x))

    def kl_div(self, lv, mu, sigma, b):
        prior_sigma = getattr(self, "{}_prior_sigma".format(lv))
        xent = ((((b**2).sum(-1) if b is not None else 0) + sigma + mu**2) / prior_sigma).sum()
        eye = torch.eye(self.rank_cov).cuda() if mu.is_cuda else torch.eye(self.rank_cov)
        ent = - math.log(prior_sigma) + torch.log(sigma).sum() + mu.numel() + \
            (torch.logdet(torch.matmul(b.t() * (1. / (sigma)), b) + eye) if b is not None else 0)
        return 0.5 * (xent - ent)

    def sample_weight(self, lv, mu, sigma, b, sample, calculate_log_probs):
        if sample:
            epsilon = self.normal.sample(sigma.shape)
            epsilon = epsilon.cuda() if mu.is_cuda else epsilon
            if b is not None:
                zeta = self.normal.sample((b.shape[1], 1))
                zeta = zeta.cuda() if mu.is_cuda else zeta
            weight = mu + sigma * epsilon + (torch.matmul(b, zeta).squeeze() if b is not None else 0)
        else:
            weight = mu

        kl = self.kl_div(lv, mu, sigma, b) if calculate_log_probs else 0.
        return weight, kl

    def forward(self, input, task, task_idx, language, sample, calculate_log_probs):
        lang_var = self.softrelu(self.lang_logvar)
        task_var = self.softrelu(self.task_logvar)
        lang_fct = self.softrelu(self.lang_fctvar)
        task_fct = self.softrelu(self.task_fctvar)

        all_lang_emb, kl_l = self.sample_weight("lang", self.lang_mean, lang_var, lang_fct, sample, calculate_log_probs)
        all_task_emb, kl_t = self.sample_weight("task", self.task_mean, task_var, task_fct, sample, calculate_log_probs)

        lang_emb = all_lang_emb[language * self.emb_dim: (language+1) * self.emb_dim]
        task_emb = all_task_emb[task_idx * self.emb_dim: (task_idx+1) * self.emb_dim]

        emb = torch.cat((lang_emb, task_emb, lang_emb - task_emb, lang_emb * task_emb), -1)
        emb = self.MLP_in(emb)
        for lyr in range(2, self.n_layers):
            MLP = getattr(self, "MLP_{}".format(lyr))
            emb = MLP(emb)
        emb = self.MLP_out(emb)
        emb2weight_mean = getattr(self, "emb2weight_mean_{}".format(task))
        emb2weight_logvar = getattr(self, "emb2weight_logvar_{}".format(task))
        emb2bias = getattr(self, "emb2bias_{}".format(task))
        weight_mean = emb2weight_mean(emb)
        weight_var = self.softrelu(emb2weight_logvar(emb))
        weight, kl_w = self.sample_weight("theta", weight_mean, weight_var, None, sample, False)
        weight = weight.view(self.n_classes[task], -1)
        bias = emb2bias(emb).view(self.n_classes[task]).squeeze()
        logits = F.linear(input, weight, bias)
        kl = kl_l + kl_t + kl_w
        return logits, kl


class MultiTaskBert(BertPreTrainedModel):

    def __init__(self, config, n_classes, mode, languages, emb_dim, n_samples, num_hidden_layers, rank_cov, largest_source):
        super(MultiTaskBert, self).__init__(config)
        self.n_classes = n_classes
        self.mode = mode
        self.languages = languages
        self.n_samples = n_samples
        assert mode in ['transfer', 'svimeta', 'lrcmeta']

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.mode == 'transfer':
            for task, n_c in n_classes.items():
                for language in self.languages if not largest_source else ['en']:
                    self.add_module('classifier_{}_{}'.format(task, language), nn.Linear(config.hidden_size, n_c))
        elif self.mode in ['svimeta', 'lrcmeta']:
            self.lang2idx = {lang: idx for idx, lang in enumerate(sorted(self.languages))}
            self.task2idx = {task: idx for idx, task in enumerate(sorted(self.n_classes.keys()))}
            if self.mode == 'svimeta':
                self.classifier = SVIMetaNet(len(languages), n_classes, config.hidden_size, emb_dim, config.hidden_dropout_prob,
                                             n_layers=num_hidden_layers)
            elif self.mode == 'lrcmeta':
                self.classifier = LRCMetaNet(len(languages), n_classes, config.hidden_size, emb_dim, config.hidden_dropout_prob,
                                             n_layers=num_hidden_layers, rank_cov=rank_cov)
        self.loss_fct = CrossEntropyLoss()
        self.apply(self.init_weights)

    def forward(self, bert_batch, task, language, sample=True, calculate_log_probs=True):

        bert_ids, labels, lengths = bert_batch
        max_length = lengths[:, 1 if task == "nli" else None].max().item()
        if max_length < bert_ids.shape[1]:
            bert_ids = bert_ids[:, :max_length]
            labels = labels[:, :max_length] if task != 'nli' else labels
        attention_mask = (bert_ids != 0)
        token_type_ids = torch.zeros_like(bert_ids)
        if task == 'nli':
            for s in range(lengths.shape[0]):
                start, end = lengths[s]
                token_type_ids[s, start:end] = 1

        outputs = self.bert(bert_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        if task == 'nli':
            output = self.dropout(outputs[1])
        else:
            output = self.dropout(outputs[0][(labels != -1).to(torch.bool)])
            labels = labels[(labels != -1).to(torch.bool)]

        logits, kl_term = 0., 0.
        if self.mode == 'transfer':
            classifier = getattr(self, 'classifier_{}_{}'.format(task, language))
            logits = classifier(output)
            loss = self.loss_fct(logits.view(-1, self.n_classes[task]), labels.view(-1))
        elif self.mode in ['svimeta', 'lrcmeta']:
            language_idx = torch.LongTensor([self.lang2idx[language]]).to(output.device)
            task_idx = torch.LongTensor([self.task2idx[task]]).to(output.device)
            n_samples = self.n_samples if sample else 1
            for n in range(1, n_samples+1):
                sample_logits, sample_kl = self.classifier(output, task, task_idx, language_idx, sample, calculate_log_probs)
                # Incremental averaging
                logits = logits + (1. / n) * (sample_logits - logits)
                kl_term = kl_term + (1. / n) * (sample_kl - kl_term)
            loss = self.loss_fct(logits.view(-1, self.n_classes[task]), labels.view(-1))

        loss = loss / len(labels) if task != "nli" else loss
        return loss, logits, kl_term
