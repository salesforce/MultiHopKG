#!/usr/bin/env python3

"""
 Copyright (c) 2018, salesforce.com, inc.
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Experiment Portal.
"""

import copy
import itertools
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
import logging

from torch.nn.utils import clip_grad_norm_
# From transformers import general tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm
import argparse

import multihopkg.data_utils as data_utils
import multihopkg.eval
from multihopkg.emb.emb import EmbeddingBasedMethod
from multihopkg.emb.fact_network import (
    ComplEx,
    ConvE,
    DistMult,
    get_complex_kg_state_dict,
    get_conve_kg_state_dict,
    get_distmult_kg_state_dict,
)
from multihopkg.hyperparameter_range import hp_range
from multihopkg.knowledge_graph import KnowledgeGraph

# LG: This immediately parses things. A script basically.
from multihopkg.learn_framework import LFramework
from multihopkg.run_configs import alpha
from multihopkg.rl.graph_search.pg import PolicyGradient
from multihopkg.rl.graph_search.pn import GraphSearchPolicy
from multihopkg.rl.graph_search.rs_pg import RewardShapingPolicyGradient
from multihopkg.utils.ops import flatten
from multihopkg.utils.convenience import not_implemented
from multihopkg.logging import setup_logger
from multihopkg.utils.setup import set_seeds
from multihopkg.models.construction import (
    construct_env_model,
    construct_navagent,
    construct_embedding_model,
)
from typing import Any, Dict, Tuple
from multihopkg.utils.ops import int_fill_var_cuda, var_cuda, zeros_var_cuda



def initialize_model_directory(args, random_seed=None):
    # add model parameter info to model directory
    # TODO: We might2ant our implementation of something like this later
    raise NotImplementedError


def construct_models(args):
    """
    Will load or construct the models for the experiment
    """
    # TODO: Get the model constructed well
    raise NotImplementedError
    models = {
        "GraphEmbedding": None,  # One of: Distmult, Complex, Conve
        "PolicyGradient": None,
        "RewardShapingPolicyGradient": None,
    }

    # for model_name, model in models.items():
    #     if model_name == "GraphEmbedding":
    #         model = construct_graph_embedding_model(args)
    #     elif model_name == "PolicyGradient":
    #         model = construct_policy_gradient_model(args)
    #     elif model_name == "RewardShapingPolicyGradient":
    #         model = construct_reward_shaping_policy_gradient_model(args)
    #     else:
    #         raise NotImplementedError



# TODO: re-implement this ?
# def inference(lf):
# ... ( you can find it in ./multihopkg/experiments.py )



def initial_setup() -> Tuple[argparse.Namespace, PreTrainedTokenizer, logging.Logger]:
    global logger
    args = alpha.get_args()
    torch.cuda.set_device(args.gpu)
    set_seeds(args.seed)
    logger = setup_logger("__MAIN__")

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    assert isinstance(args, argparse.Namespace)

    return args, tokenizer, logger



def losses_fn(mini_batch):
    # TODO:
    raise NotImplementedError
    # def stablize_reward(r):
    #     r_2D = r.view(-1, self.num_rollouts)
    #     if self.baseline == 'avg_reward':
    #         stabled_r_2D = r_2D - r_2D.mean(dim=1, keepdim=True)
    #     elif self.baseline == 'avg_reward_normalized':
    #         stabled_r_2D = (r_2D - r_2D.mean(dim=1, keepdim=True)) / (r_2D.std(dim=1, keepdim=True) + ops.EPSILON)
    #     else:
    #         raise ValueError('Unrecognized baseline function: {}'.format(self.baseline))
    #     stabled_r = stabled_r_2D.view(-1)
    #     return stabled_r
    #
    # e1, e2, r = self.format_batch(mini_batch, num_tiles=self.num_rollouts)
    # output = self.rollout(e1, r, e2, num_steps=self.num_rollout_steps)
    #
    # # Compute policy gradient loss
    # pred_e2 = output['pred_e2']
    # log_action_probs = output['log_action_probs']
    # action_entropy = output['action_entropy']
    #
    # # Compute discounted reward
    # final_reward = self.reward_fun(e1, r, e2, pred_e2)
    # if self.baseline != 'n/a':
    #     final_reward = stablize_reward(final_reward)
    # cum_discounted_rewards = [0] * self.num_rollout_steps
    # cum_discounted_rewards[-1] = final_reward
    # R = 0
    # for i in range(self.num_rollout_steps - 1, -1, -1):
    #     R = self.gamma * R + cum_discounted_rewards[i]
    #     cum_discounted_rewards[i] = R
    #
    # # Compute policy gradient
    # pg_loss, pt_loss = 0, 0
    # for i in range(self.num_rollout_steps):
    #     log_action_prob = log_action_probs[i]
    #     pg_loss += -cum_discounted_rewards[i] * log_action_prob
    #     pt_loss += -cum_discounted_rewards[i] * torch.exp(log_action_prob)
    #
    # # Entropy regularization
    # entropy = torch.cat([x.unsqueeze(1) for x in action_entropy], dim=1).mean(dim=1)
    # pg_loss = (pg_loss - entropy * self.beta).mean()
    # pt_loss = (pt_loss - entropy * self.beta).mean()
    #
    # loss_dict = {}
    # loss_dict['model_loss'] = pg_loss
    # loss_dict['print_loss'] = float(pt_loss)
    # loss_dict['reward'] = final_reward
    # loss_dict['entropy'] = float(entropy.mean())
    # if self.run_analysis:
    #     fn = torch.zeros(final_reward.size())
    #     for i in range(len(final_reward)):
    #         if not final_reward[i]:
    #             if int(pred_e2[i]) in self.kg.all_objects[int(e1[i])][int(r[i])]:
    #                 fn[i] = 1
    #     loss_dict['fn'] = fn
    #
    # return loss_dict

# TODO: Finish this inner training loop.
def batch_train(
    batch_size: int,
    grad_norm: float,
    model: nn.Module,
    optimizer: torch.optim.Optimizer, # type: ignore
    train_data: torch.Tensor,
) -> Dict[str,Any]:

    # TODO: Decide on the batch metrics.
    batch_metrics = {
        "loss": [],
        "entropy": [],
    }

    for sample_offset_idx in tqdm(range(0, len(train_data), batch_size)):
        optimizer.zero_grad()
        mini_batch = train_data[sample_offset_idx:sample_offset_idx + batch_size]
        if len(mini_batch) < batch_size:
            continue

        loss = losses_fn(mini_batch)
        loss['model_loss'].backward()
        if grad_norm > 0:
            clip_grad_norm_(model.parameters(), grad_norm)

        optimizer.step()

        batch_metrics["loss"].append(loss['print_loss'])
        if 'entropy' in loss:
            batch_metrics["entropy"].append(loss['entropy'])


        # TODO: Need to figure out what `run_analysis` is doing and whether we want it
        # TOREM: If unecessary
        # if self.run_analysis:
        #     if rewards is None:
        #         rewards = loss['reward']
        #     else:
        #         rewards = torch.cat([rewards, loss['reward']])
        #     if fns is None:
        #         fns = loss['fn']
        #     else:
        #         fns = torch.cat([fns, loss['fn']])
    return batch_metrics


def train_multihopkg(
    batch_size: int,
    epochs: int,
    fmodel: nn.Module,
    grad_norm: float,
    # knowledge_graph: KnowledgeGraph,
    learning_rate: float,
    start_epoch: int,
):

    # Print Model Parameters + Perhaps some more information
    print('Model Parameters')
    print('--------------------------')
    for name, param in fmodel.named_parameters():
        print(name, param.numel(), 'requires_grad={}'.format(param.requires_grad))

    # Just use Adam Optimizer by defailt
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, fmodel.parameters()), lr=learning_rate
    )  # type:ignore

    #TODO: Metrics to track
    metrics_to_track = {'loss', 'entropy'}
    for epoch_id in range(start_epoch, epochs):
        logger.info('Epoch {}'.format(epoch_id))

        # TODO: Perhaps evaluate the epochs?

        # Set in training mode
        fmodel.train()
    
        # TOREM: Perhapas no need for this shuffle.
        # random.shuffle(train_data)
        batch_losses = []
        entropies = []

        # TODO: Understand if this is actually necessary here
        # if self.run_analysis:
        #     rewards = None
        #     fns = None

        ##############################
        # Batch Iteration Starts Here.
        ##############################
        # TODO: update the parameters.
        batch_metrics = batch_train(
            batch_size,
            grad_norm,
            fmodel,
            optimizer,
            train_data,
        )

        # Check training statistics
        stdout_msg = 'Epoch {}: average training loss = {}'.format(epoch_id, np.mean(batch_losses))
        if entropies:
            stdout_msg += ' entropy = {}'.format(np.mean(entropies))
        print(stdout_msg)
        self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id)
        if self.run_analysis:
            print('* Analysis: # path types seen = {}'.format(self.num_path_types))
            num_hits = float(rewards.sum())
            hit_ratio = num_hits / len(rewards)
            print('* Analysis: # hits = {} ({})'.format(num_hits, hit_ratio))
            num_fns = float(fns.sum())
            fn_ratio = num_fns / len(fns)
            print('* Analysis: false negative ratio = {}'.format(fn_ratio))

        # Check dev set performance
        if self.run_analysis or (epoch_id > 0 and epoch_id % self.num_peek_epochs == 0):
            self.eval()
            self.batch_size = self.dev_batch_size
            dev_scores = self.forward(dev_data, verbose=False)
            print('Dev set performance: (correct evaluation)')
            _, _, _, _, mrr = src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.dev_objects, verbose=True)
            metrics = mrr
            print('Dev set performance: (include test set labels)')
            src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.all_objects, verbose=True)
            # Action dropout anneaking
            if self.model.startswith('point'):
                eta = self.action_dropout_anneal_interval
                if len(dev_metrics_history) > eta and metrics < min(dev_metrics_history[-eta:]):
                    old_action_dropout_rate = self.action_dropout_rate
                    self.action_dropout_rate *= self.action_dropout_anneal_factor 
                    print('Decreasing action dropout rate: {} -> {}'.format(
                        old_action_dropout_rate, self.action_dropout_rate))
            # Save checkpoint
            if metrics > best_dev_metrics:
                self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id, is_best=True)
                best_dev_metrics = metrics
                with open(os.path.join(self.model_dir, 'best_dev_iteration.dat'), 'w') as o_f:
                    o_f.write('{}'.format(epoch_id))
            else:
                # Early stopping
                if epoch_id >= self.num_wait_epochs and metrics < np.mean(dev_metrics_history[-self.num_wait_epochs:]):
                    break
            dev_metrics_history.append(metrics)
            if self.run_analysis:
                num_path_types_file = os.path.join(self.model_dir, 'num_path_types.dat')
                dev_metrics_file = os.path.join(self.model_dir, 'dev_metrics.dat')
                hit_ratio_file = os.path.join(self.model_dir, 'hit_ratio.dat')
                fn_ratio_file = os.path.join(self.model_dir, 'fn_ratio.dat')
                if epoch_id == 0:
                    with open(num_path_types_file, 'w') as o_f:
                        o_f.write('{}\n'.format(self.num_path_types))
                    with open(dev_metrics_file, 'w') as o_f:
                        o_f.write('{}\n'.format(metrics))
                    with open(hit_ratio_file, 'w') as o_f:
                        o_f.write('{}\n'.format(hit_ratio))
                    with open(fn_ratio_file, 'w') as o_f:
                        o_f.write('{}\n'.format(fn_ratio))
                else:
                    with open(num_path_types_file, 'a') as o_f:
                        o_f.write('{}\n'.format(self.num_path_types))
                    with open(dev_metrics_file, 'a') as o_f:
                        o_f.write('{}\n'.format(metrics))
                    with open(hit_ratio_file, 'a') as o_f:
                        o_f.write('{}\n'.format(hit_ratio))
                    with open(fn_ratio_file, 'a') as o_f:
                        o_f.write('{}\n'.format(fn_ratio))


def rollout(
    # TODO: self.mdl should point to (policy network)
    kg: KnowledgeGraph,
    num_steps,
    pn: PolicyGradient,
    policy_network: GraphSearchPolicy,
    query: torch.Tensor,
    visualize_action_probs=False,
):
    assert (num_steps > 0)

    # Initialization
    # TOREM: Figure out how to get the dimension of the relationships and embeddings 
    entity_shape = not_implemented(torch.tensor, "entity_shape","mlm_training.py::rollout()")

    # These are all very reinforcement-learning things
    log_action_probs = []
    action_entropy = []

    # Dummy nodes ? TODO: Figur eout what they do.
    # TODO: Perhaps here we can enter through the centroid.
    # For now we still with these dummy
    r_s = int_fill_var_cuda(entity_shape, kg.dummy_start_r)
    # NOTE: We repeat these entities until we get the right shape:
    seen_nodes = int_fill_var_cuda(entity_shape, kg.dummy_e).unsqueeze(1)
    path_components = []

    # Save some history
    path_trace = [(r_s, e_s)]
    # NOTE:(LG): Must be run as `.reset()` for ensuring environment `pn` is stup
    pn.initialize_path((r_s, e_s), kg)

    for t in range(num_steps):
        last_r, e = path_trace[-1]
        obs = [e_s, q, e_t, t==(num_steps-1), last_r, seen_nodes]

        db_outcomes, inv_offset, policy_entropy = pn.transit(
            e, obs, kg, use_action_space_bucketing=self.use_action_space_bucketing)

        sample_outcome = self.sample_action(db_outcomes, inv_offset)
        action = sample_outcome['action_sample']
        pn.update_path(action, kg)
        action_prob = sample_outcome['action_prob']
        log_action_probs.append(ops.safe_log(action_prob))
        action_entropy.append(policy_entropy)
        seen_nodes = torch.cat([seen_nodes, e.unsqueeze(1)], dim=1)
        path_trace.append(action)

        if visualize_action_probs:
            top_k_action = sample_outcome['top_actions']
            top_k_action_prob = sample_outcome['top_action_probs']
            path_components.append((e, top_k_action, top_k_action_prob))

    pred_e2 = path_trace[-1][1]
    self.record_path_trace(path_trace)

    return {
        'pred_e2': pred_e2,
        'log_action_probs': log_action_probs,
        'action_entropy': action_entropy,
        'path_trace': path_trace,
        'path_components': path_components
    }

def main():
    # By default we run the config
    # Process data will determine by itself if there is any data to process
    args, tokenizer, logger = initial_setup()

    # TODO: Muybe ? (They use it themselves)
    # initialize_model_directory(args, args.seed)

    # Setting up the models
    logger.info(":: (1/3) Loaded embedding model")
    env = GraphSearchPolicy(
        relation_only=args.relation_only,
        history_dim=args.history_dim,
        history_num_layers=args.history_num_layers,
        entity_dim=args.entity_dim,
        relation_dim=args.relation_dim,
        ff_dropout_rate=args.ff_dropout_rate,
        xavier_initialization=args.xavier_initialization,
        relation_only_in_path=args.relation_only_in_path,
    )
    logger.info(":: (2/3) Loaded environment module")

    ## Agent needs a Knowledge graph as well as the environment
    knowledge_graph = KnowledgeGraph(
        bandwidth = args.bandwidth,
        data_dir = args.data_dir,
        model = args.model,
        entity_dim = args.entity_dim,
        relation_dim = args.relation_dim,
        emb_dropout_rate = args.emb_dropout_rate,
        num_graph_convolution_layers = args.num_graph_convolution_layers,
        use_action_space_bucketing = args.use_action_space_bucketing,
        bucket_interval = args.bucket_interval,
        test = args.test,
        relation_only = args.relation_only,
    )

    nav_agent = PolicyGradient(
        args.use_action_space_bucketing,
        args.num_rollouts,
        args.baseline,
        args.beta,
        args.gamma,
        args.action_dropout_rate,
        args.action_dropout_anneal_factor,
        args.action_dropout_anneal_interval,
        args.beam_size,
        knowledge_graph,
        env, # What you just created above
        args.num_rollout_steps,
        args.model_dir,
        args.model,
        args.data_dir,
        args.batch_size,
        args.train_batch_size,
        args.dev_batch_size,
        args.start_epoch,
        args.num_epochs,
        args.num_wait_epochs,
        args.num_peek_epochs,
        args.learning_rate,
        args.grad_norm,
        args.adam_beta1,
        args.adam_beta2,
        args.train,
        args.run_analysis,
    )

    logger.info(":: (3/3) Loaded navigation agent")
    logger.info(":: Training the model")


    # TODO: Add checkpoint support:
    start_epoch = 0

    ######## ######## ########
    # Train:
    ######## ######## ########
    entity_index_path = os.path.join(args.data_dir, "entity2id.txt")
    relation_index_path = os.path.join(args.data_dir, "relation2id.txt")

    text_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    train_data, metadata = data_utils.process_qa_data(
        args.raw_QAPathData_path,
        args.cached_QAPathData_path,
        text_tokenizer,
    )
    list_train_data = list(train_data.values)
    

    # TODO: Load the validation data
    # dev_path = os.path.join(args.data_dir, "dev.triples")
    # dev_data = data_utils.load_triples(
    #     dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities
    # )


    # TODO: Make it take check for a checkpoint and decide what start_epoch
    # if args.checkpoint_path is not None:
    #     # TODO: Add it here to load the checkpoint separetely
    #     nav_agent.load_checkpoint(args.checkpoint_path)
    start_epoch = 0
    dev_data = None


    train_multihopkg(
        args.batch_size,
        args.epochs,
        nav_agent,
        args.grad_norm,
        knowledge_graph,
        args.learning_rate,
        args.start_epoch,
        start_epoch,
        list_train_data,
    )
    # lf.run_train(train_data, dev_data)

    # TODO: Evaluation of the model
    # metrics = inference(lf)


if __name__ == "__main__":
    main()
