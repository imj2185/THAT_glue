# coding=utf-8
# Length-Adaptive Transformer
# Copyright (c) 2020-present NAVER Corp.
# Apache License v2.0
#####
# Original code is from https://github.com/huggingface/transformers
# Copyright 2020-present the HuggingFace Inc. team.
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
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import csv
import os
import random
import math
import timeit
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchprofile
from packaging import version
import torch.nn.functional as F
from torch import nn
from torch.nn import KLDivLoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange

from transformers.file_utils import is_torch_tpu_available
from transformers.integrations import (
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalPrediction,
    EvaluationStrategy,
    HPSearchBackend,
    PredictionOutput,
    distributed_broadcast_scalars,
    distributed_concat,
    nested_concat,
    nested_numpify,
    nested_xla_mesh_reduce,
    set_seed,
)
from transformers.utils import logging
from transformers import Trainer
from transformers import PreTrainedTokenizer


_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        from tensorboardX import SummaryWriter

if is_wandb_available():
    import wandb

if is_comet_available():
    import comet_ml

if is_optuna_available():
    import optuna

if is_ray_available():
    from ray import tune


from token_head_adaptive_transformer.drop_and_restore_utils import (
    LengthDropArguments,
    sample_length_configuration,
    sample_layer_configuration,
)
from token_head_adaptive_transformer.evolution import store2str

logger = logging.get_logger(__name__)

def sample_head_configuration(
    num_heads,
    num_hidden_layers,
    layer_config=None,
    max_head_pruning=False,
    random_head_pruning=False,
    min_head=1,
):
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

class LengthDropTrainer(Trainer):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer = None,
        best_metric: str = 'acc',
        length_drop_args: LengthDropArguments = None,
        **kwargs,
    ):
        super(LengthDropTrainer, self).__init__(**kwargs)
        self.tokenizer = tokenizer
        self.best_metric = best_metric
        if length_drop_args is None:
            length_drop_args = LengthDropArguments()
        self.length_drop_args = length_drop_args

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            retention_params = []
            wd_params = []
            no_wd_params = []
            for n, p in self.model.named_parameters():
                if "retention" in n:
                    retention_params.append(p)
                elif any(nd in n for nd in no_decay):
                    no_wd_params.append(p)
                else:
                    wd_params.append(p)
            optimizer_grouped_parameters = [
                {"params": wd_params, "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate},
                {"params": no_wd_params, "weight_decay": 0.0, "lr": self.args.learning_rate}
            ]
            if len(retention_params) > 0:
                optimizer_grouped_parameters.append(
                    {"params": retention_params, "weight_decay": 0.0, "lr": self.length_drop_args.lr_soft_extract}
                )
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if self.lr_scheduler is None:
            if self.args.warmup_ratio is not None:
                num_warmup_steps = int(self.args.warmup_ratio * num_training_steps)
            else:
                num_warmup_steps = self.args.warmup_steps
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
            )

    def div_loss(self, loss):
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        return loss

    def train(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
        """
        Main training entry point.
        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)
            model = self.model_init()
            self.model = model.to(self.args.device)

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(num_update_steps_per_epoch * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
            self.args.max_steps = t_total

        self.create_optimizer_and_scheduler(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        if self.args.fp16 and _use_apex:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split(os.path.sep)[0])

                epochs_trained = self.global_step // num_update_steps_per_epoch
                steps_trained_in_current_epoch = self.global_step % (num_update_steps_per_epoch)

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss_sum = 0.0
        loss_sum = defaultdict(float)
        best = {self.best_metric: None} if self.best_metric != 'pearson_and_spearman' else {'corr': None}
        model.zero_grad()
        disable_tqdm = self.args.disable_tqdm or not self.is_local_process_zero()
        train_pbar = trange(epochs_trained, int(np.ceil(num_train_epochs)), desc="Epoch", disable=disable_tqdm)
        
        if "distilbert" in model_path:
            bert = model.module.distilbert if hasattr(model, "module") else model.distilbert
        elif "roberta" in model_path:
            bert = model.module.roberta if hasattr(model, "module") else model.roberta
        elif "mobilebert" in model_path:
            bert = model.module.mobilebert if hasattr(model, "module") else model.mobilebert
        else:
            bert = model.module.bert if hasattr(model, "module") else model.bert

        hi_path_list = model_path.split(os.sep)
        head_importance_path = os.path.join(hi_path_list[0], hi_path_list[1], hi_path_list[2], "head_importance.pt")
        head_importance = torch.zeros(model.config.num_hidden_layers, model.config.num_attention_heads).to(bert.device)
        
        for step, inputs in enumerate(train_dataloader):
            model.train()
            inputs = self._prepare_inputs(inputs)

            inputs["output_attentions"] = None
            inputs["layer_config"] = None
            inputs["length_config"] = None
            inputs["head_prune"] = True

            outputs = model(**inputs)
            task_loss = self.div_loss(outputs[0])
            loss = task_loss
            loss.backward()

            for layer in range(model.config.num_hidden_layers):
                self_att = bert.transformer.layer[layer].attention if "distilbert" in model_path else bert.encoder.layer[layer].attention.self
                ctx = self_att.context_layer_val
                grad_ctx = ctx.grad
                # Take the dot
                dot = torch.einsum("bhli,bhli->bhl", [grad_ctx, ctx])
                head_importance[layer] += dot.abs().sum(-1).sum(0).detach()

            tot_tokens = inputs["attention_mask"].float().detach().sum().data
        head_importance /= tot_tokens

        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1/exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

        print("head_importance_measured!")
        print(head_importance)

        torch.save(head_importance, head_importance_path)

        model.zero_grad()
        for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            epoch_pbar = tqdm(epoch_iterator, desc="Iteration", disable=disable_tqdm)
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    epoch_pbar.update(1)
                    continue

                model.train()
                inputs = self._prepare_inputs(inputs)

                inputs["output_attentions"] = self.length_drop_args.length_config is not None

                layer_config = sample_layer_configuration(
                    model.module.config.num_hidden_layers if self.args.n_gpu > 1 else model.config.num_hidden_layers,
                    layer_dropout_prob=self.length_drop_args.layer_dropout_prob,
                    layer_dropout=0,
                )
                inputs["layer_config"] = layer_config
                inputs["head_prune"] = True

                inputs["length_config"] = self.length_drop_args.length_config

                outputs = model(**inputs)
                # Save past state if it exists
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index]
                task_loss = self.div_loss(outputs[0])
                if self.length_drop_args.length_adaptive:
                    loss_sum["full"] += task_loss.item()
                loss = task_loss
                if self.length_drop_args.length_adaptive:
                    loss = loss / (self.length_drop_args.num_sandwich + 2)

                tr_loss_sum += loss.item()
                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # inplace distillation
                if self.length_drop_args.length_adaptive:
                    logits = outputs[1].detach()

                    for i in range(self.length_drop_args.num_sandwich + 1):
                        inputs["output_attentions"] = True

                        layer_config = sample_layer_configuration(
                            model.module.config.num_hidden_layers if self.args.n_gpu > 1 else model.config.num_hidden_layers,
                            layer_dropout_prob=self.length_drop_args.layer_dropout_prob,
                            layer_dropout=(self.length_drop_args.layer_dropout_bound if i == 0 else None),
                            layer_dropout_bound=self.length_drop_args.layer_dropout_bound,
                        )
                        inputs["layer_config"] = layer_config

                        length_config = sample_length_configuration(
                            self.args.max_seq_length,
                            model.module.config.num_hidden_layers if self.args.n_gpu > 1 else model.config.num_hidden_layers,
                            layer_config,
                            length_drop_ratio=(self.length_drop_args.length_drop_ratio_bound if i == 0 else None),
                            length_drop_ratio_bound=self.length_drop_args.length_drop_ratio_bound,
                        )
                        inputs["length_config"] = length_config

                        head_config = sample_head_configuration(
                        bert.config.num_attention_heads,
                        bert.config.num_hidden_layers,
                        layer_config,
                        max_head_pruning= (True if i == 0 else False),
                        random_head_pruning=True,
                        )

                        inputs["head_config"] = what_to_prune(
                            head_importance, 
                            head_config, 
                            to_prune={},
                            at_least_x_heads_per_layer=1,
                        )

                        outputs_sub = model(**inputs)
                        task_loss_sub = self.div_loss(outputs_sub[0])
                        if i == 0:
                            loss_sum["smallest"] += task_loss_sub.item()
                            loss_sum["sub"] += 0
                        else:
                            loss_sum["sub"] += task_loss_sub.item() / self.length_drop_args.num_sandwich

                        logits_sub = outputs_sub[1]
                        loss_fct = KLDivLoss(reduction="batchmean")
                        kl_loss = loss_fct(F.log_softmax(logits, -1), F.softmax(logits_sub, -1))
                        loss = self.div_loss(kl_loss)
                        loss_sum["kl"] += loss.item() / (self.length_drop_args.num_sandwich + 1)
                        loss = loss / (self.length_drop_args.num_sandwich + 2)

                        tr_loss_sum += loss.item()
                        if self.args.fp16 and _use_native_amp:
                            self.scaler.scale(loss).backward()
                        elif self.args.fp16 and _use_apex:
                            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    (step + 1) == len(epoch_iterator) <= self.args.gradient_accumulation_steps
                ):
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        # backward compatibility for pytorch schedulers
                        lr = (
                            self.lr_scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else self.lr_scheduler.get_lr()[0]
                        )
                        loss = tr_loss_sum / self.args.logging_steps
                        tr_loss_sum = 0.0
                        logs = {"lr": lr, "loss": loss}
                        log_str = f"[{self.global_step:5d}] lr {lr:g} | loss {loss:2.3f}"

                        for key, value in loss_sum.items():
                            value /= self.args.logging_steps
                            loss_sum[key] = 0.0
                            logs[f"{key}_loss"] = value
                            log_str += f" | {key}_loss {value:2.3f}"

                        self.log(logs, "train")
                        logger.info(log_str)

                    '''
                    if (
                        self.args.evaluation_strategy == EvaluationStrategy.STEPS
                        and self.global_step % self.args.eval_steps == 0
                    ):
                        results = self.evaluate()
                        self._report_to_hp_search(trial, epoch, results)
                    '''

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert (
                                model.module is self.model
                            ), f"Module {model.module} should be a reference to self.model"
                        else:
                            assert model is self.model, f"Model {model} should be a reference to self.model"

                        if self.args.evaluate_during_training:
                            results = self.evaluate()
                            results = {k[5:]: v for k, v in results.items() if k.startswith("eval_")}
                            self.log(results, "dev")
                            msg = " | ".join([f"{k} {v:.3f}" for k, v in results.items()])
                            logger.info(f"  [{self.global_step:5d}] {msg}")

                        # Save model checkpoint
                        if self.args.save_only_best:
                            output_dirs = []
                        else:
                            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}"
                            if self.hp_search_backend is not None and trial is not None:
                                run_id = (
                                    trial.number
                                    if self.hp_search_backend == HPSearchBackend.OPTUNA
                                    else tune.get_trial_id()
                                )
                                checkpoint_folder += f"-run-{run_id}"
                            output_dirs = [os.path.join(self.args.output_dir, checkpoint_folder)]
                            
                        if self.args.evaluate_during_training:
                            if self.best_metric == "pearson_and_spearman":
                                if best['corr'] is None or results['corr'] > best['corr']:
                                    logger.info("Congratulations, best model so far!")
                                    output_dirs.append(os.path.join(self.args.output_dir, "checkpoint-best"))
                                    best = results
                            else:
                                if best[self.best_metric] is None or results[self.best_metric] > best[self.best_metric]:
                                    logger.info("Congratulations, best model so far!")
                                    output_dirs.append(os.path.join(self.args.output_dir, "checkpoint-best"))
                                    best = results

                        for output_dir in output_dirs:
                            self.save_model(output_dir)

                            if self.is_world_master() and self.tokenizer is not None:
                                self.tokenizer.save_pretrained(output_dir)

                            if self.is_world_process_zero():
                                self._rotate_checkpoints(use_mtime=True)

                            '''
                            if is_torch_tpu_available():
                                xm.rendezvous("saving_optimizer_states")
                                xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            elif self.is_world_process_zero():
                                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            '''

                epoch_pbar.update(1)
                if 0 < self.args.max_steps <= self.global_step:
                    break
            epoch_pbar.close()
            train_pbar.update(1)

            '''
            if self.args.evaluation_strategy == EvaluationStrategy.EPOCH:
                results = self.evaluate()
                self._report_to_hp_search(trial, epoch, results)
            '''

            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if 0 < self.args.max_steps <= self.global_step:
                break

        train_pbar.close()
        if self.tb_writer:
            self.tb_writer.close()
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return self.global_step, best

    def log(self, logs, mode="train"):
        self._setup_loggers()
        if self.global_step is None:
            # when logging evaluation metrics without training
            self.global_step = 0
        if self.tb_writer:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, self.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        '"%s" of type %s for key "%s" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute.",
                        v,
                        type(v),
                        k,
                    )
            self.tb_writer.flush()
        if is_wandb_available():
            if self.is_world_process_zero():
                wandb.log(logs, step=self.global_step)
        if is_comet_available():
            if self.is_world_process_zero():
                experiment = comet_ml.config.get_global_experiment()
                if experiment is not None:
                    experiment._log_metrics(logs, step=self.global_step, epoch=self.epoch, framework="transformers")
        output = {**logs, **{"step": self.global_step}}
        if self.is_world_process_zero():
            self.log_history.append(output)

    def evaluate(self, eval_dataset: Optional[Dataset] = None, head_prune: Optional[bool] = None, evo_search: Optional[bool] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output, latency = self.prediction_loop(eval_dataloader, description="Evaluation", head_prune=head_prune)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        if evo_search is not None:
            return output.metrics, latency
        else:
            return output.metrics

    def get_head_importance(self, eval_dataset=None, model_type=None):
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        head_importance = self.head_importance_loop(eval_dataloader, description="Evaluation", model_type=model_type)

        return head_importance

    def head_importance_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None, model_type: str = None,
    ):

        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        '''
        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        '''
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        if "distilbert" in model_type:
            bert = model.module.distilbert if hasattr(model, "module") else model.distilbert
        elif "roberta" in model_type:
            bert = model.module.roberta if hasattr(model, "module") else model.roberta
        elif "mobilebert" in model_type:
            bert = model.module.mobilebert if hasattr(model, "module") else model.mobilebert
        else:
            bert = model.module.bert if hasattr(model, "module") else model.bert

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm

        head_importance = torch.zeros(model.config.num_hidden_layers, model.config.num_attention_heads).to(model.device)
        tot_tokens = 0

        for inputs in tqdm(dataloader, desc=description, disable=disable_tqdm):
            loss = self.head_importance_step(model, inputs, prediction_loss_only)
            batch_size = inputs[list(inputs.keys())[0]].shape[0]

            loss.backward()

            for layer in range(model.config.num_hidden_layers):
                self_att = bert.transformer.layer[layer].attention if "distilbert" in model_type else bert.encoder.layer[layer].attention.self
                ctx = self_att.context_layer_val
                grad_ctx = ctx.grad
                # Take the dot
                dot = torch.einsum("bhli,bhli->bhl", [grad_ctx, ctx])
                head_importance[layer] += dot.abs().sum(-1).sum(0).detach()

            tot_tokens += inputs["attention_mask"].float().detach().sum().data
        head_importance /= tot_tokens

        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1/exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

        return head_importance
            
    def head_importance_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool
    ):
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.args.label_names)
        inputs = self._prepare_inputs(inputs)

        output_attentions = getattr(inputs, 'output_attentions', None)
        output_hidden_states = getattr(inputs, 'output_hidden_states', None)

        # if self.args.n_gpu > 1:
        #     config_output_attentions = self.model.module.config.output_attentions
        #     config_output_hidden_states = self.model.module.config.output_hidden_states
        # else:
        #     config_output_attentions = self.model.config.output_attentions
        #     config_output_hidden_states = self.model.config.output_hidden_states

        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        
        outputs = model(**inputs)
        if has_labels:
            # The .mean() is to reduce in case of distributed training
            loss = outputs[0]
        
        return loss

    def prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None, head_prune: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if hasattr(self, "_prediction_loop"):
            warnings.warn(
                "The `_prediction_loop` method is deprecated and won't be called in a future version, define `prediction_loop` in your subclass.",
                FutureWarning,
            )
            return self._prediction_loop(dataloader, description, prediction_loss_only=prediction_loss_only)

        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        '''
        assert not getattr(
            self.model.config, "output_attentions", False
        ), "The prediction loop does not work with `output_attentions=True`."
        assert not getattr(
            self.model.config, "output_hidden_states", False
        ), "The prediction loop does not work with `output_hidden_states=True`."
        '''

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        '''
        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        '''
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm
        start_time = timeit.default_timer()
        for inputs in tqdm(dataloader, desc=description, disable=disable_tqdm):
            if head_prune is None:
                inputs["head_prune"] = True
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only)
            batch_size = inputs[list(inputs.keys())[0]].shape[0]
            if loss is not None:
                eval_losses.extend([loss] * batch_size)
            if logits is not None:
                preds = logits if preds is None else nested_concat(preds, logits, dim=0)
            if labels is not None:
                label_ids = labels if label_ids is None else nested_concat(label_ids, labels, dim=0)
        evalTime = timeit.default_timer() - start_time

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
        elif is_torch_tpu_available():
            # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
            if preds is not None:
                preds = nested_xla_mesh_reduce(preds, "eval_preds")
            if label_ids is not None:
                label_ids = nested_xla_mesh_reduce(label_ids, "eval_label_ids")
            if eval_losses is not None:
                eval_losses = xm.mesh_reduce("eval_losses", torch.tensor(eval_losses), torch.cat).tolist()

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = nested_numpify(preds)
        if label_ids is not None:
            label_ids = nested_numpify(label_ids)

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            if self.args.local_rank != -1:
                metrics["eval_loss"] = (
                    distributed_broadcast_scalars(eval_losses, num_total_examples=self.num_examples(dataloader))
                    .mean()
                    .item()
                )
            else:
                metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics), evalTime

    def prediction_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.args.label_names)
        inputs = self._prepare_inputs(inputs)

        output_attentions = getattr(inputs, 'output_attentions', None)
        output_hidden_states = getattr(inputs, 'output_hidden_states', None)

        # if self.args.n_gpu > 1:
        #     config_output_attentions = self.model.module.config.output_attentions
        #     config_output_hidden_states = self.model.module.config.output_hidden_states
        # else:
        #     config_output_attentions = self.model.config.output_attentions
        #     config_output_hidden_states = self.model.config.output_hidden_states

        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        num_additional_outputs = int(output_attentions == True) + int(output_hidden_states == True)

        
        outputs = model(**inputs)
        if has_labels:
            # The .mean() is to reduce in case of distributed training
            loss = outputs[0].mean().item()
            logits = outputs[1:(len(outputs) - num_additional_outputs)]
        else:
            loss = None
            # Slicing so we get a tuple even if `outputs` is a `ModelOutput`.
            logits = outputs[:(len(outputs) - num_additional_outputs)]
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = tuple(logit.detach() for logit in logits)
        if len(logits) == 1:
            logits = logits[0]

        if has_labels:
            labels = tuple(inputs.get(name).detach() for name in self.args.label_names)
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        return (loss, logits, labels)

    def init_evolution(self, lower_constraint=0, upper_constraint=None):
        size = (1, self.args.max_seq_length)
        self.dummy_inputs = (
            torch.ones(size, dtype=torch.long).to(self.args.device),
            torch.ones(size, dtype=torch.long).to(self.args.device),
            torch.zeros(size, dtype=torch.long).to(self.args.device),
        )
        if self.model.config.model_type == "distilbert":
            self.dummy_inputs = self.dummy_inputs[:2]


        self.lower_constraint = lower_constraint
        self.upper_constraint = upper_constraint

        self.store = {}  # gene: (macs, score, method, parent(s))
        self.population = []

    def load_store(self, store_file):
        if not os.path.isfile(store_file):
            return
        with open(store_file, 'r') as f:
            for row in csv.reader(f, delimiter='\t'):
                row = tuple(eval(x) for x in row[:3])
                self.store[row[0]] = row[1:3] + (0, None)

    def save_store(self, store_file):
        store_keys = sorted(self.store.keys(), key=lambda x: self.store[x][0])
        with open(store_file, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            for gene in store_keys:
                writer.writerow([str(gene)] + [str(x) for x in self.store[gene]])

    def save_population(self, population_file, population):
        with open(population_file, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            for gene in population:
                writer.writerow([str(gene)] + [str(x) for x in self.store[gene]])

    def ccw(self, gene0, gene1, gene2):
        x0, y0 = self.store[gene0][:2]
        x1, y1 = self.store[gene1][:2]
        x2, y2 = self.store[gene2][:2]
        return (x0 * y1 + x1 * y2 + x2 * y0) - (x0 * y2 + x1 * y0 + x2 * y1)

    def convex_hull(self):
        hull = self.population[:2]
        for gene in self.population[2:]:
            if self.store[hull[-1]][1] >= self.store[gene][1]:
                continue
            while len(hull) >= 2 and self.ccw(hull[-2], hull[-1], gene) >= 0:
                del hull[-1]
            hull.append(gene)
        return hull

    def pareto_frontier(self):
        self.population = sorted(self.population, key=lambda x: self.store[x][:2])

        frontier = [self.population[0]]
        for gene in self.population[1:-1]:
            if self.store[gene][1] > self.store[frontier[-1]][1]:
                if self.store[gene][0] == frontier[-1][0]:
                    del frontier[-1]
                frontier.append(gene)
        frontier.append(self.population[-1])
        self.population = frontier

        area = 0
        for gene0, gene1 in zip(self.population[:-1], self.population[1:]):
            x0, y0 = self.store[gene0][:2]
            x1, y1 = self.store[gene1][:2]
            area += (x1 - x0) * y0
        area /= (self.upper_constraint - self.lower_constraint)
        return self.population, area

    def add_gene(self, gene, method=0, parents=None, latency=False):
        if gene not in self.store:
            self.model.eval()
            if self.model.config.model_type == "distilbert":
                bert = self.model.distilbert
            elif self.model.config.model_type == "mobilebert":
                bert = self.model.mobilebert
            else:
                assert hasattr(self.model, "bert")
                bert = self.model.bert
            bert.set_length_config(gene[0])
            hi_path_list = self.args.output_dir.split(os.sep)
            head_importance_path = os.path.join(hi_path_list[0], hi_path_list[1], hi_path_list[2], "head_importance.pt")
            #head_importance = self.get_head_importance(model_type=self.model.config.model_type)
            head_importance = torch.load(head_importance_path)

            to_prune = what_to_prune(
                    head_importance,
                    gene[1],
                    to_prune={},
                    at_least_x_heads_per_layer=1,    
            )
            bert.prune_heads(to_prune)
            if not latency:
                macs_or_latency = torchprofile.profile_macs(self.model, args=self.dummy_inputs)
                # logger.info(gene, macs)
                if macs_or_latency < self.lower_constraint:
                    bert.revert_pruned_heads(to_prune)
                    return False
                metric, _ = self.evaluate(head_prune=False, evo_search=True)
            else:
                metric, macs_or_latency = self.evaluate(head_prune=False, evo_search=True)
            bert.revert_pruned_heads(to_prune)

            metric_name = "eval_" + self.best_metric if self.best_metric != 'pearson_and_spearman' else "eval_corr"
            score = metric[metric_name]
            self.store[gene] = (macs_or_latency, score, method, parents)
            logger.info(store2str(gene, macs_or_latency, score, method, parents))

        macs_or_latency = self.store[gene][0]
        if macs_or_latency >= self.lower_constraint \
                and (self.upper_constraint is None or macs_or_latency <= self.upper_constraint) \
                and gene not in self.population:
            self.population.append(gene)
            return True
        return False

    def mutate(self, mutation_prob, latency_constraint):
        gene = random.choice(self.population)
        mutated_gene_length = ()
        mutated_gene_head = ()
        for i in range(self.model.config.num_hidden_layers):
            if np.random.uniform() < mutation_prob:
                prev = (self.args.max_seq_length if i == 0 else mutated_gene_length[i - 1])
                next = (2 if i == self.model.config.num_hidden_layers - 1 else gene[0][i + 1])
                mutated_gene_length += (random.randrange(next, prev + 1),)
                
                prev = (0 if i == 0 else mutated_gene_head[i - 1])
                next = (self.model.config.num_attention_heads - 1 if i == self.model.config.num_hidden_layers - 1 else gene[1][i + 1])
                mutated_gene_head += (random.randrange(prev, next + 1),)
            else:
                mutated_gene_length += (gene[0][i],)
                mutated_gene_head += (gene[1][i],)

        mutated_gene = (mutated_gene_length, mutated_gene_head)
        return self.add_gene(mutated_gene, method=1, parents=(gene,), latency=latency_constraint)

    def crossover(self, latency_constraint):
        gene0, gene1 = random.sample(self.population, 2)
        crossovered_gene_length = tuple((g0 + g1 + 1) // 2 for g0, g1 in zip(gene0[0], gene1[0]))
        crossovered_gene_head = tuple((g0 + g1 + 1) // 2 for g0, g1 in zip(gene0[1], gene1[1]))
        crossovered_gene = (crossovered_gene_length, crossovered_gene_head)
        return self.add_gene(crossovered_gene, method=2, parents=(gene0, gene1), latency=latency_constraint)
