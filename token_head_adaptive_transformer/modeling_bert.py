# coding=utf-8
# Length-Adaptive Transformer
# Copyright (c) 2020-present NAVER Corp.
# Apache License v2.0
#####
# Original code is from https://github.com/huggingface/transformers
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model. """


import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_callable,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    SequenceClassifierOutput,
    QuestionAnsweringModelOutput,
)
from transformers.modeling_utils import apply_chunking_to_forward
from transformers.modeling_bert import (
    _CONFIG_FOR_DOC,
    _TOKENIZER_FOR_DOC,
    BertEmbeddings,
    BertIntermediate,
    BertOutput,
    BertPooler,
    BertPreTrainedModel,
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING,
)

from token_head_adaptive_transformer.modeling_utils import expand_gather
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

BertLayerNorm = torch.nn.LayerNorm

def copy_linear_layer(source_layer, target_layer):
    W = source_layer.weight.clone().detach()
    if source_layer.bias is not None:
        b = source_layer.bias.clone().detach()

    target_layer.weight.requires_grad = False
    target_layer.weight.copy_(W.contiguous())
    target_layer.weight.requires_grad = True

    if source_layer.bias is not None:
        target_layer.bias.requires_grad = False
        target_layer.bias.copy_(b.contiguous())
        target_layer.bias.requires_grad = True

def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking :obj:`already_pruned_heads` into account.

    Args:
        heads (:obj:`List[int]`): List of the indices of heads to prune.
        n_heads (:obj:`int`): The number of heads in the model.
        head_size (:obj:`int`): The size of each head.
        already_pruned_heads (:obj:`Set[int]`): A set of already pruned heads.

    Returns:
        :obj:`Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index

def prune_linear_layer(layer: torch.nn.Linear, index: torch.LongTensor, dim: int = 0) -> torch.nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (:obj:`torch.nn.Linear`): The layer to prune.
        index (:obj:`torch.LongTensor`): The indices to keep in the layer.
        dim (:obj:`int`, `optional`, defaults to 0): The dimension on which to keep the indices.

    Returns:
        :obj:`torch.nn.Linear`: The pruned layer as a new layer with :obj:`requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer

def revert_pruned_linear_layer(base_layer) -> torch.nn.Linear:
    """
    Revert a linear layer back to the original entries before the pruning.

    Used to recover heads.

    Args:
        base_layer (:obj:`torch.nn.Linear`): The layer to recover.

    Returns:
        :obj:`torch.nn.Linear`: The recovered layer as a new layer with :obj:`requires_grad=True`.
    """
    W = base_layer.weight.clone().detach()
    if base_layer.bias is not None:
        b = base_layer.bias.clone().detach()

    new_size = list(base_layer.weight.size())
    new_layer = torch.nn.Linear(new_size[1], new_size[0], bias=base_layer.bias is not None).to(base_layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if base_layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


class LinearSuper(nn.Linear):
    def __init__(self, super_in_dim, super_out_dim, bias=True, uniform_=None, non_linear='linear'):
        super().__init__(super_in_dim, super_out_dim, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        self._reset_parameters(bias, uniform_, non_linear)

    def _reset_parameters(self, bias, uniform_, non_linear):
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(
            self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.)

    def forward(self, x, index=None, dim=0):
        if index is not None:
            if dim == 0:
                return F.linear(x, self.weight[index,:], self.bias[index])
            else:
                return F.linear(x, self.weight[:,index], self.bias)
        else:
            return F.linear(x, self.weight, self.bias)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.original_num_attention_heads = config.num_attention_heads
        self.original_attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.original_all_head_size = self.original_num_attention_heads * self.original_attention_head_size

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.pruned_heads = set()

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.query_f = LinearSuper(config.hidden_size, self.all_head_size)
        self.key_f = LinearSuper(config.hidden_size, self.all_head_size)
        self.value_f = LinearSuper(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        head_prune=False,
        head_index=None,
    ):
        if head_prune:
            mixed_query_layer = self.query_f(hidden_states, head_index)
        else:
            mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            if head_prune:
                mixed_key_layer = self.key_f(encoder_hidden_states, head_index)
                mixed_value_layer = self.value_f(encoder_hidden_states, head_index)
            else:
                mixed_key_layer = self.key(encoder_hidden_states)
                mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            if head_prune:
                mixed_key_layer = self.key_f(hidden_states, head_index)
                mixed_value_layer = self.value_f(hidden_states, head_index)
            else:
                mixed_key_layer = self.key(hidden_states)
                mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
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
        self.context_layer_val = context_layer
        self.context_layer_val.retain_grad()

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_f = LinearSuper(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, head_prune=False, head_index=None):
        if head_prune:
            hidden_states = self.dense_f(hidden_states, head_index, dim=1)
        else:
            hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def set_fn_layer_parameters(self):
        copy_linear_layer(self.self.query, self.self.query_f)
        copy_linear_layer(self.self.key, self.self.key_f)
        copy_linear_layer(self.self.value, self.self.value_f)
        copy_linear_layer(self.output.dense, self.output.dense_f)

        # self.self.query_f.set_sample_config()
        # self.self.key_f.set_sample_config()
        # self.self.value_f.set_sample_config()
        # self.output.dense_f.set_sample_config()

    def set_nn_layer_parameters(self):
        copy_linear_layer(self.self.query_f, self.self.query)
        copy_linear_layer(self.self.key_f, self.self.key)
        copy_linear_layer(self.self.value_f, self.self.value)
        copy_linear_layer(self.output.dense_f, self.output.dense)

    def set_sample_config(self, heads):
        self.self.num_attention_heads = self.self.original_num_attention_heads
        self.self.all_head_size = self.self.original_all_head_size
        self.pruned_heads = set()

        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        self.self.query_f.set_sample_config(index, prune=True)
        self.self.key_f.set_sample_config(index, prune=True)
        self.self.value_f.set_sample_config(index, prune=True)
        self.output.dense_f.set_sample_config(index, dim=1, prune=True)

        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def revert_heads(self, heads, other=None):
        # Prune linear layers
        self.self.query = revert_pruned_linear_layer(self.self.query_f)
        self.self.key = revert_pruned_linear_layer(self.self.key_f)
        self.self.value = revert_pruned_linear_layer(self.self.value_f)
        self.output.dense = revert_pruned_linear_layer(self.output.dense_f)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads + len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        head_prune=False,
        heads=None,
    ):
        if heads is not None:
            heads, head_index = find_pruneable_heads_and_indices(
                heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
            )

            self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
            self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
            self.pruned_heads = self.pruned_heads.union(heads)

        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            head_prune=head_prune,
            head_index=head_index if heads is not None else None,
        )
        attention_output = self.output(self_outputs[0], hidden_states, head_prune=head_prune, head_index=head_index if heads is not None else None,)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        if heads is not None:
            self.self.num_attention_heads = self.self.original_num_attention_heads
            self.self.all_head_size = self.self.original_attention_head_size * self.self.original_num_attention_heads
            self.pruned_heads = set()
            
        return outputs


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_length=None,
        always_keep_cls_token=True,
        head_prune=False,
        head_index=None
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            head_prune=head_prune,
            heads=head_index,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        if output_length is not None:
            assert output_attentions
            attention_probs = self_attention_outputs[1]
            significance_score = attention_probs.sum(2).sum(1)  # - attention_probs.diagonal(0, 2, 3)
            if always_keep_cls_token:
                keep_indices = significance_score[:, 1:].topk(output_length - 1, 1)[1] + 1
                cls_index = keep_indices.new_zeros((keep_indices.size(0), 1))
                keep_indices = torch.cat((cls_index, keep_indices), 1)
            else:
                keep_indices = significance_score.topk(output_length, 1)[1]
            # keep_indices = keep_indices.sort(1)[0]
            attention_output = expand_gather(attention_output, 1, keep_indices.unsqueeze(-1))
        else:
            keep_indices = None

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs, keep_indices

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        layer_config=None,
        length_config=None,
        head_config=None,
        always_keep_cls_token=True,
        head_prune=False,
    ):
        bsz, tsz, dim = hidden_states.size()

        if length_config is not None:
            restored_hidden_states = hidden_states
            remain_indices = torch.arange(tsz, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)

        all_hidden_states = () if output_hidden_states else None
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if layer_config is not None and i not in layer_config:
                continue

            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_output_length = length_config[i] if length_config is not None else None
            layer_head_prune = head_config[i] if head_config is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions, layer_output_length, always_keep_cls_token)

                    return custom_forward

                layer_outputs, keep_indices = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs, keep_indices = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                    output_length=layer_output_length,
                    always_keep_cls_token=always_keep_cls_token,
                    head_prune=head_prune,
                    head_index=layer_head_prune
                )
            hidden_states = layer_outputs[0]

            if layer_output_length:
                remain_indices = remain_indices.gather(1, keep_indices)
                restored_hidden_states = restored_hidden_states.scatter(1, remain_indices.unsqueeze(-1).expand(-1, -1, dim), hidden_states)

                if attention_mask is not None:
                    attention_mask = expand_gather(attention_mask, 3, keep_indices.unsqueeze(1).unsqueeze(2))
                    if attention_mask.size(2) > 1:
                        attention_mask = expand_gather(attention_mask, 2, keep_indices.unsqueeze(1).unsqueeze(3))

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        last_hidden_state = restored_hidden_states if length_config is not None else hidden_states
        if not return_dict:
            return tuple(v for v in [last_hidden_state, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=last_hidden_state, hidden_states=all_hidden_states, attentions=all_attentions
        )


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

        self.length_config = None
        self.head_config = None

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def set_length_config(self, length_config):
        self.length_config = length_config

    def set_head_config(self, head_config):
        self.head_config = head_config

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )

    def set_sample_config(self, to_mask):
        for layer in range(len(self.encoder.layer)):
            if layer in to_mask:
                heads = to_mask[layer]
                att = self.encoder.layer[layer].attention
                att.set_sample_config(heads)

    def set_fn_layer_parameters(self):
        for layer in range(len(self.encoder.layer)):
            att = self.encoder.layer[layer].attention
            att.set_fn_layer_parameters()

    def set_nn_layer_parameters(self):
        for layer in range(len(self.encoder.layer)):
            att = self.encoder.layer[layer].attention
            att.set_nn_layer_parameters()

    def mask_heads(self, to_mask):
        for layer in range(len(self.encoder.layer)):
            if layer in to_mask:
                heads = to_mask[layer]
                self_att = self.encoder.layer[layer].attention.self
                self_att.mask_heads = list(heads)
                self_att._head_mask = None

    def prune_heads(self, to_mask):
        for layer in range(len(self.encoder.layer)):
            if layer in to_mask:
                heads = to_mask[layer]
                att = self.encoder.layer[layer].attention
                att.prune_heads(heads)

    def revert_pruned_heads(self, to_mask, other=None):
        for layer in range(len(self.encoder.layer)):
            if layer in to_mask:
                heads = to_mask[layer]
                att = self.encoder.layer[layer].attention
                att.revert_heads(heads, other)

    def clear_heads_mask(self):
        for layer in range(len(self.encoder.layer)):
            self_att = self.encoder.layer[layer].attention.self
            self_att.mask_heads = []
            self_att._head_mask = None

    def mask_heads_grad(self, to_mask):
        for layer, heads in to_mask.items():
            self.encoder.layer[layer].attention.mask_heads_grad(heads)

    def reset_heads(self, to_reset, other_bert=None):
        other_layer = None
        for layer, heads in to_reset.items():
            if other_bert is not None:
                other_layer = other_bert.encoder.layer[layer].attention
            self.encoder.layer[layer].attention.reset_heads(heads, other_layer)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        layer_config=None,
        length_config=None,
        head_config=None,
        always_keep_cls_token=True,
        head_prune=False,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            layer_config=layer_config,
            length_config=length_config if length_config is not None else self.length_config,
            head_config=head_config if head_config is not None else self.head_config,
            always_keep_cls_token=always_keep_cls_token,
            head_prune=head_prune,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    BERT_START_DOCSTRING,
)
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )

    def set_sample_config(self, head_config):
        self.bert.set_sample_config(head_config)

    def set_fn_layer_parameters(self):
        self.bert.set_fn_layer_parameters()

    def set_nn_layer_parameters(self):
        self.bert.set_nn_layer_parameters()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        layer_config=None,
        length_config=None,
        head_config=None,
        always_keep_cls_token=True,
        head_prune=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            layer_config=layer_config,
            length_config=length_config,
            head_config=head_config,
            always_keep_cls_token=always_keep_cls_token,
            head_prune=head_prune,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`). """,
    BERT_START_DOCSTRING,
)
class BertForQuestionAnswering(BertPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )

    def set_sample_config(self, head_config):
        self.bert.set_sample_config(head_config)

    def set_fn_layer_parameters(self):
        self.bert.set_fn_layer_parameters()

    def set_nn_layer_parameters(self):
        self.bert.set_nn_layer_parameters()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        layer_config=None,
        length_config=None,
        head_config=None,
        always_keep_cls_token=False,
        head_prune=False,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            layer_config=layer_config,
            length_config=length_config,
            head_config=head_config,
            always_keep_cls_token=always_keep_cls_token,
            head_prune=head_prune,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
