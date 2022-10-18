# Length-Adaptive Transformer
# Copyright (c) 2020-present NAVER Corp.
# Apache License v2.0

from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import math


def sample_length_configuration(
    max_seq_length,
    num_hidden_layers,
    layer_config=None,
    length_drop_prob=None,
    length_drop_ratio=None,
    length_drop_ratio_bound=None,
    min_length=2,
):
    length = max_seq_length
    length_configuration = ()
    for i in range(num_hidden_layers):
        if layer_config is None or i in layer_config:
            if length_drop_prob is not None:
                length = length - np.random.binomial(length, length_drop_prob)
            elif length_drop_ratio is not None:
                length = int(np.ceil(length * (1 - length_drop_ratio)))
            elif length_drop_ratio_bound is not None:
                length = np.random.randint(int(np.ceil(length * (1 - length_drop_ratio_bound))), length + 1)
        length = max(length, min_length)
        length_configuration += (length,)
    return length_configuration


def sample_layer_configuration(
    num_hidden_layers,
    layer_dropout_prob=None,
    layer_dropout=None,
    layer_dropout_bound=None,
):
    if layer_dropout_prob is not None:
        return tuple(i for i in range(num_hidden_layers) if np.random.random() >= layer_dropout_prob)
    elif layer_dropout is not None:
        layer_dropout = min(layer_dropout, num_hidden_layers - 1)
        return tuple(range(num_hidden_layers - layer_dropout))
    elif layer_dropout_bound is not None:
        layer_dropout_bound = min(layer_dropout_bound, num_hidden_layers - 1)
        return tuple(range(num_hidden_layers - np.random.randint(0, layer_dropout_bound + 1)))
    return None


def sample_head_configuration(
    num_heads,
    num_hidden_layers,
    layer_config=None,
    max_head_pruning=False,
    random_head_pruning=False,
    min_head=1,
    prune_ratio=None,
):
    if prune_ratio is not None:
        max_pruning_configuration = [math.floor(r) for r in np.linspace(min_head, prune_ratio, num_hidden_layers)]
    else:
        max_pruning_configuration = [math.floor(r) for r in np.linspace(min_head, num_heads, num_hidden_layers)]
        max_pruning_configuration[-1] -= 1
    
    head = 0
    head_configuration = ()
    for i in range(num_hidden_layers):
        if layer_config is None or i in layer_config:
            if max_head_pruning:
                head = max_pruning_configuration[i]
            elif random_head_pruning:
                head = np.random.randint(head, max_pruning_configuration[i])
        head_configuration += (head,)
    return head_configuration

def what_to_prune(
    head_importance,
    gene,
    to_prune=None,
    at_least_x_heads_per_layer=0,
    rescale_by_number=False,
):
    head_importance = head_importance.clone()
    n_layers, n_heads = head_importance.size()
    to_prune = to_prune or {}
    if rescale_by_number:
        for layer in to_prune:
            #head_importance[layer] *= sqrt(n_layers / len(to_prune[layer]))
            head_importance[layer] *= math.sqrt(len(to_prune[layer]) / n_layers)
    # Sort heads by score
    heads_and_score = [
        ((layer, head), head_importance[layer, head])
        for layer in range(n_layers)
        for head in range(n_heads)
    ]
    heads_and_score = sorted(heads_and_score, key=lambda x: x[1])
    sorted_heads = [head_and_score[0]
                    for head_and_score in heads_and_score]
    # Ensure we don't delete all heads in a layer
    if at_least_x_heads_per_layer:
        # Remove the top scoring head in each layer
        to_protect = {l: 0 for l in range(n_layers)}
        filtered_sorted_heads = []
        for layer, head in reversed(sorted_heads):
            if layer in to_protect:
                if to_protect[layer] < at_least_x_heads_per_layer:
                    to_protect[layer] += 1
                    continue
                else:
                    to_protect.pop(layer)
            filtered_sorted_heads.insert(0, (layer, head))
        sorted_heads = filtered_sorted_heads
    # layer/heads that were already pruned
    # Prune the lowest scoring heads
    sorted_heads = [
        (layer, head)
        for (layer, head) in sorted_heads
        if layer not in to_prune or head not in to_prune[layer]
    ]
    # Update heads to prune
    for layer, head in sorted_heads:
        if layer not in to_prune:
            to_prune[layer] = []
        if len(to_prune[layer]) < gene[layer]:
            to_prune[layer].append(head)
    return to_prune

@dataclass
class LengthDropArguments:
    length_config: Optional[List[int]] = None
    length_adaptive: Optional[bool] = False
    num_sandwich: Optional[int] = 2
    length_drop_ratio_bound: Optional[float] = 0.2
    layer_dropout_prob: Optional[float] = 0.2
    layer_dropout_bound: Optional[int] = 0


def add_drop_and_restore_args(parser):
    parser.add_argument(
        "--length_config",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--length_adaptive",
        action="store_true",
    )
    parser.add_argument(
        "--num_sandwich",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--length_drop_ratio_bound",
        default=0.2,
        type=float,
    )
    parser.add_argument(
        "--layer_dropout_prob",
        default=None,
        type=float,
    )
    parser.add_argument(
        "--layer_dropout_bound",
        default=0,
        type=int,
    )


@dataclass
class SearchArguments:
    do_search: Optional[bool] = field(default=False)
    do_ray_search: Optional[bool] = field(default=False)
    latency_constraint: Optional[bool] = field(default=False)
    load_store_file: Optional[str] = field(default=None)
    evo_iter: Optional[int] = field(default=100)
    population_size: Optional[int] = field(default=20)
    mutation_size: Optional[int] = field(default=30)
    mutation_prob: Optional[float] = field(default=0.5)
    crossover_size: Optional[int] = field(default=30)


def add_search_args(parser):
    parser.add_argument("--do_search", action="store_true")
    parser.add_argument("--latency_constraint", action="store_true")
    parser.add_argument("--load_store_file", default=None, type=str)
    parser.add_argument("--evo_iter", default=100, type=int)
    parser.add_argument("--population_size", default=20, type=int)
    parser.add_argument("--mutation_size", default=30, type=int)
    parser.add_argument("--mutation_prob", default=0.5, type=float)
    parser.add_argument("--crossover_size", default=30, type=int)
