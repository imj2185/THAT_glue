# coding=utf-8
# Length-Adaptive Transformer
# Copyright (c) 2020-present NAVER Corp.
# Apache License v2.0
#####
# Original code is from https://github.com/huggingface/transformers
# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
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
""" PyTorch DistilBERT model
    adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
    and in part from HuggingFace PyTorch version of Google AI Bert model (https://github.com/google-research/bert)
"""


import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_callable,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    SequenceClassifierOutput,
    QuestionAnsweringModelOutput,
)
from transformers.modeling_distilbert import (
    _CONFIG_FOR_DOC,
    _TOKENIZER_FOR_DOC,
    Embeddings,
    FFN,
    DistilBertPreTrainedModel,
    DISTILBERT_START_DOCSTRING,
    DISTILBERT_INPUTS_DOCSTRING,
)

from token_head_adaptive_transformer.modeling_utils import expand_gather
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

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

def revert_pruned_linear_layer(base_layer: torch.nn.Linear) -> torch.nn.Linear:
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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.attention_head_size = int(config.dim / config.n_heads)
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)

        self.original_n_heads = config.n_heads
        self.original_dim = config.dim

        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)

        self.q_lin_f = LinearSuper(config.dim, config.dim)
        self.k_lin_f = LinearSuper(config.dim, config.dim)
        self.v_lin_f = LinearSuper(config.dim, config.dim)
        self.out_lin_f = LinearSuper(config.dim, config.dim)

        self.pruned_heads = set()

    def set_fn_layer_parameters(self):
        copy_linear_layer(self.q_lin, self.q_lin_f)
        copy_linear_layer(self.k_lin, self.k_lin_f)
        copy_linear_layer(self.v_lin, self.v_lin_f)
        copy_linear_layer(self.out_lin, self.out_lin_f)

        # self.self.query_f.set_sample_config()
        # self.self.key_f.set_sample_config()
        # self.self.value_f.set_sample_config()
        # self.output.dense_f.set_sample_config()

    def set_nn_layer_parameters(self):
        copy_linear_layer(self.q_lin_f, self.q_lin)
        copy_linear_layer(self.k_lin_f, self.k_lin)
        copy_linear_layer(self.v_lin_f, self.v_lin)
        copy_linear_layer(self.out_lin_f, self.out_lin)

    def prune_heads(self, heads):
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, attention_head_size, self.pruned_heads)
        # Prune linear layers
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def revert_heads(self, heads, other=None):
        attention_head_size = self.dim // self.n_heads
        # Prune linear layers
        self.q_lin = revert_pruned_linear_layer(self.q_lin_f)
        self.k_lin = revert_pruned_linear_layer(self.k_lin_f)
        self.v_lin = revert_pruned_linear_layer(self.v_lin_f)
        self.out_lin = revert_pruned_linear_layer(self.out_lin_f)

        # Update hyper params and store pruned heads
        self.n_heads = self.n_heads + len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = set()

    def forward(self, query, key, value, mask, head_mask=None, output_attentions=False, head_prune=False, head_index=None):
        """
        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        # assert key.size() == value.size()

        #dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, self.attention_head_size).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.attention_head_size)

        if head_prune:
            q = shape(self.q_lin_f(query, head_index))  # (bs, n_heads, q_length, dim_per_head)
            k = shape(self.k_lin_f(key, head_index))  # (bs, n_heads, k_length, dim_per_head)
            v = shape(self.v_lin_f(value, head_index))  # (bs, n_heads, k_length, dim_per_head)
        else:
            q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
            k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
            v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(self.attention_head_size)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        self.context_layer_val = context
        self.context_layer_val.retain_grad()

        context = unshape(context)  # (bs, q_length, dim)
        if head_prune:
            context = self.out_lin_f(context, head_index, dim=1)  # (bs, q_length, dim)
        else:
            context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.dim % config.n_heads == 0

        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_length=None,
        always_keep_cls_token=True,
        head_prune=False,
        heads=None,
    ):
        """
        Parameters
        ----------
        hidden_states: torch.tensor(bs, seq_length, dim)
        attention_mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        attention_probs: torch.tensor(bs, n_heads, seq_length, seq_length)
            The attention weights
        layer_output: torch.tensor(bs, seq_length, dim)
            The output of the transformer block contextualization.
        """
        if heads is not None:
            heads, head_index = find_pruneable_heads_and_indices(
                heads, self.attention.n_heads, self.attention.attention_head_size, self.attention.pruned_heads
            )

            self.attention.n_heads = self.attention.n_heads - len(heads)
            self.attention.dim = self.attention.attention_head_size * self.attention.n_heads
            self.attention.pruned_heads = self.attention.pruned_heads.union(heads)

        # Self-Attention
        self_attention_outputs = self.attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            mask=attention_mask,
            head_mask=head_mask,
            output_attentions = output_attentions,
            head_prune=head_prune,
            head_index=head_index if heads is not None else None,
        )
        if output_attentions:
            attention_output, attention_probs = self_attention_outputs  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attention` or `output_hidden_states` cases returning tuples
            assert type(self_attention_outputs) == tuple
            attention_output = self_attention_outputs[0]
        attention_output = self.sa_layer_norm(attention_output + hidden_states)  # (bs, seq_length, dim)

        if output_length is not None:
            assert output_attentions
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

        # Feed Forward Network
        layer_output = self.ffn(attention_output)  # (bs, seq_length, dim)
        layer_output = self.output_layer_norm(layer_output + attention_output)  # (bs, seq_length, dim)

        output = (layer_output,)
        if output_attentions:
            output = (attention_probs,) + output

        if heads is not None:
            self.attention.n_heads = self.attention.original_n_heads
            self.attention.dim = self.attention.attention_head_size * self.attention.original_n_heads
            self.pruned_heads = set()

        return output, keep_indices


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_hidden_layers = config.n_layers
        self.head_importance = None
        self.layer = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=None,
        layer_config=None,
        length_config=None,
        head_config=None,
        always_keep_cls_token=True,
        head_prune=False,
    ):
        """
        Parameters
        ----------
        hidden_states: torch.tensor(bs, seq_length, dim)
            Input sequence embedded.
        attention_mask: torch.tensor(bs, seq_length)
            Attention mask on the sequence.

        Outputs
        -------
        hidden_state: torch.tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if output_attentions=True
        """
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

            layer_outputs, keep_indices = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                output_attentions,
                output_length=layer_output_length,
                always_keep_cls_token=always_keep_cls_token,
                head_prune=head_prune,
                heads=layer_head_prune,
            )
            hidden_states = layer_outputs[-1]

            if layer_output_length:
                remain_indices = remain_indices.gather(1, keep_indices)
                restored_hidden_states = restored_hidden_states.scatter(1, remain_indices.unsqueeze(-1).expand(-1, -1, dim), hidden_states)

                if attention_mask is not None:
                    attention_mask = attention_mask.gather(1, keep_indices)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

        last_hidden_state = restored_hidden_states if length_config is not None else hidden_states
        if not return_dict:
            return tuple(v for v in [last_hidden_state, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=last_hidden_state, hidden_states=all_hidden_states, attentions=all_attentions
        )


@add_start_docstrings(
    "The bare DistilBERT encoder/transformer outputting raw hidden-states without any specific head on top.",
    DISTILBERT_START_DOCSTRING,
)
class DistilBertModel(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = Embeddings(config)  # Embeddings
        self.transformer = Transformer(config)  # Encoder

        self.init_weights()

        self.length_config = None
        self.head_config = None

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.transformer.layer[layer].attention.prune_heads(heads)

    def set_length_config(self, length_config):
        self.length_config = length_config

    def set_head_config(self, head_config):
        self.head_config = head_config

    def set_fn_layer_parameters(self):
        for layer in range(len(self.transformer.layer)):
            att = self.transformer.layer[layer].attention
            att.set_fn_layer_parameters()

    def set_nn_layer_parameters(self):
        for layer in range(len(self.transformer.layer)):
            att = self.transformer.layer[layer].attention
            att.set_nn_layer_parameters()

    def prune_heads(self, to_mask):
        for layer in range(len(self.transformer.layer)):
            if layer in to_mask:
                heads = to_mask[layer]
                att = self.transformer.layer[layer].attention
                att.prune_heads(heads)

    def revert_pruned_heads(self, to_mask, other=None):
        for layer in range(len(self.transformer.layer)):
            if layer in to_mask:
                heads = to_mask[layer]
                att = self.transformer.layer[layer].attention
                att.revert_heads(heads, other)

    def set_head_importance_parameters(self):
        self.transformer.head_importance = torch.tensor([[0.3240, 0.2014, 0.2765, 0.3280, 0.3863, 0.1581, 0.1790, 0.1569, 0.3282, 0.4521, 0.3320, 0.1310],[0.2349, 0.2688, 0.2251, 0.2320, 0.1632, 0.3764, 0.5694, 0.1990, 0.1632, 0.2351, 0.2122, 0.3301],[0.2338, 0.2372, 0.3248, 0.4260, 0.1656, 0.2699, 0.1958, 0.2649, 0.2820, 0.3126, 0.3897, 0.2533],[0.4295, 0.2508, 0.2263, 0.1589, 0.2339, 0.4926, 0.1191, 0.1180, 0.3259, 0.2956, 0.3573, 0.1712],[0.0880, 0.0429, 0.1927, 0.2209, 0.0428, 0.0297, 0.3825, 0.0383, 0.6678, 0.0920, 0.2021, 0.5086],[0.0743, 0.4888, 0.4030, 0.2603, 0.5059, 0.0655, 0.1710, 0.1601, 0.2533,0.3606, 0.0610, 0.1109]]).to(self.transformer.device)

    @add_start_docstrings_to_callable(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, num_choices"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="distilbert-base-uncased",
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="distilbert-base-uncased")
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        layer_config=None,
        length_config=None,
        head_config=None,
        always_keep_cls_token=True,
        head_prune=False,
    ):
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
            attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.n_layers)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)  # (bs, seq_length, dim)
        return self.transformer(
            inputs_embeds,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            layer_config=layer_config,
            length_config=length_config if length_config is not None else self.length_config,
            head_config=head_config if head_config is not None else self.head_config,
            always_keep_cls_token=always_keep_cls_token,
            head_prune=head_prune,
        )


@add_start_docstrings(
    """DistilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    DISTILBERT_START_DOCSTRING,
)
class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()

    def set_fn_layer_parameters(self):
        self.distilbert.set_fn_layer_parameters()

    def set_nn_layer_parameters(self):
        self.distilbert.set_nn_layer_parameters()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
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

        distilbert_output = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
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
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )


@add_start_docstrings(
    """DistilBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    DISTILBERT_START_DOCSTRING,
)
class DistilBertForQuestionAnswering(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.distilbert = DistilBertModel(config)
        self.qa_outputs = nn.Linear(config.dim, config.num_labels)
        assert config.num_labels == 2
        self.dropout = nn.Dropout(config.qa_dropout)

        self.init_weights()

    @add_start_docstrings_to_callable(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, num_choices"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="distilbert-base-uncased",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )

    def set_fn_layer_parameters(self):
        self.distilbert.set_fn_layer_parameters()

    def set_nn_layer_parameters(self):
        self.distilbert.set_nn_layer_parameters()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
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

        distilbert_output = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
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
        sequence_output = distilbert_output[0]  # (bs, max_query_len, dim)

        hidden_states = self.dropout(sequence_output)  # (bs, max_query_len, dim)
        logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1)  # (bs, max_query_len)

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
            output = (start_logits, end_logits) + distilbert_output[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )
