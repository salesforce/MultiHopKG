"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Policy gradient (REINFORCE algorithm) training and inference.
"""

from typing import List
import torch

from multihopkg.knowledge_graph import KnowledgeGraph
from multihopkg.learn_framework import LFramework
import multihopkg.rl.graph_search.beam_search as search
from multihopkg.rl.graph_search.pn import GraphSearchPolicy
import multihopkg.utils.ops as ops
from multihopkg.utils.ops import int_fill_var_cuda, var_cuda, zeros_var_cuda
from transformers import BertTokenizer, BertModel
from torch import nn


# TOREM: I dont really think we need the Lfraework
# class ContinuousPolicyGradient(LFramework):
class ContinuousPolicyGradient(nn.Module):
    def __init__(self):
        # TODOHow we train 

        raise NotImplementedError("`ContinuousPolicyGradient` not yet implemented.")

    def sample_action(self, ):
        raise NotImplementedError

class PolicyGradient(LFramework):
    def __init__(
        self,
        use_action_space_bucketing: bool,
        num_rollouts: int,
        baseline: str,
        beta: float,
        gamma: float,
        action_dropout_rate: float,
        action_dropout_anneal_factor: float,
        action_dropout_anneal_interval: float,
        beam_size: int,
        kg: KnowledgeGraph,
        pn: GraphSearchPolicy,
        # Goodness this is ugly:
        num_rollout_steps: int,
        model_dir: str,
        model: str,
        data_dir: str,
        batch_size: int,
        train_batch_size: int,
        dev_batch_size: int,
        start_epoch: int,
        num_epochs: int,
        num_wait_epochs: int,
        num_peek_epochs: int,
        learning_rate: float,
        grad_norm: float,
        adam_beta1: float,
        adam_beta2: float,
        train: bool,
        run_analysis: bool,
    ):

        super(PolicyGradient, self).__init__(
            model_dir,
            model,
            data_dir,
            batch_size,
            train_batch_size,
            dev_batch_size,
            start_epoch,
            num_epochs,
            num_wait_epochs,
            num_peek_epochs,
            learning_rate,
            grad_norm,
            adam_beta1,
            adam_beta2,
            train,
            run_analysis,
            kg,
            pn,
        )

        # Training hyperparameters
        self.use_action_space_bucketing = use_action_space_bucketing
        self.num_rollouts = num_rollouts
        self.num_rollout_steps = num_rollout_steps
        self.baseline = baseline
        self.beta = beta  # entropy regularization parameter
        self.gamma = gamma  # shrinking factor
        self.action_dropout_rate = action_dropout_rate
        self.action_dropout_anneal_factor = (
            action_dropout_anneal_factor  # Used in parent
        )
        self.action_dropout_anneal_interval = (
            action_dropout_anneal_interval  # Also used by parent
        )

        # Inference hyperparameters
        self.beam_size = beam_size

        # Analysis
        self.path_types = dict()
        self.num_path_types = 0

    def reward_fun(self, e1, r, e2, pred_e2):
        # TODO: Change to the one Andres should be doing. (The soft reward)
        return (pred_e2 == e2).float()

    def loss(self, mini_batch):
        
        def stablize_reward(r):
            r_2D = r.view(-1, self.num_rollouts)
            if self.baseline == 'avg_reward':
                stabled_r_2D = r_2D - r_2D.mean(dim=1, keepdim=True)
            elif self.baseline == 'avg_reward_normalized':
                stabled_r_2D = (r_2D - r_2D.mean(dim=1, keepdim=True)) / (r_2D.std(dim=1, keepdim=True) + ops.EPSILON)
            else:
                raise ValueError('Unrecognized baseline function: {}'.format(self.baseline))
            stabled_r = stabled_r_2D.view(-1)
            return stabled_r
    
        e1, e2, r = self.format_batch(mini_batch, num_tiles=self.num_rollouts)
        output = self.rollout(e1, r, e2, num_steps=self.num_rollout_steps)

        # Compute policy gradient loss
        pred_e2 = output['pred_e2']
        log_action_probs = output['log_action_probs']
        action_entropy = output['action_entropy']

        # Compute discounted reward
        final_reward = self.reward_fun(e1, r, e2, pred_e2)
        if self.baseline != 'n/a':
            final_reward = stablize_reward(final_reward)
        cum_discounted_rewards = [0] * self.num_rollout_steps
        cum_discounted_rewards[-1] = final_reward
        R = 0
        for i in range(self.num_rollout_steps - 1, -1, -1):
            R = self.gamma * R + cum_discounted_rewards[i]
            cum_discounted_rewards[i] = R

        # Compute policy gradient
        pg_loss, pt_loss = 0, 0
        for i in range(self.num_rollout_steps):
            log_action_prob = log_action_probs[i]
            pg_loss += -cum_discounted_rewards[i] * log_action_prob
            pt_loss += -cum_discounted_rewards[i] * torch.exp(log_action_prob)

        # Entropy regularization
        entropy = torch.cat([x.unsqueeze(1) for x in action_entropy], dim=1).mean(dim=1)
        pg_loss = (pg_loss - entropy * self.beta).mean()
        pt_loss = (pt_loss - entropy * self.beta).mean()

        loss_dict = {}
        loss_dict['model_loss'] = pg_loss
        loss_dict['print_loss'] = float(pt_loss)
        loss_dict['reward'] = final_reward
        loss_dict['entropy'] = float(entropy.mean())
        if self.run_analysis:
            fn = torch.zeros(final_reward.size())
            for i in range(len(final_reward)):
                if not final_reward[i]:
                    if int(pred_e2[i]) in self.kg.all_objects[int(e1[i])][int(r[i])]:
                        fn[i] = 1
            loss_dict['fn'] = fn

        return loss_dict

    # LGN: Roll out as in: "Roll out an entire episode.
    def rollout(self, e_s, q, e_t, num_steps, visualize_action_probs=False):
        """
        Perform multi-step rollout from the source entity conditioned on the query relation.
        :param pn: Policy network.
        :param e_s: (Variable:batch) source entity indices.
        :param q: (Variable:batch) query relation indices.
        :param e_t: (Variable:batch) target entity indices.
        :param kg: Knowledge graph environment.
        :param num_steps: Number of rollout steps.
        :param visualize_action_probs: If set, save action probabilities for visualization.

        :return pred_e2: Target entities reached at the end of rollout.
        :return log_path_prob: Log probability of the sampled path.
        :return action_entropy: Entropy regularization term.
        """
        assert (num_steps > 0)
        kg, pn = self.kg, self.mdl

        # Initialization

        # These are all very reinforcement-learning things
        log_action_probs = []
        action_entropy = []

        # Dummy nodes ? TODO: Figur eout what they do.
        r_s = int_fill_var_cuda(e_s.size(), kg.dummy_start_r)
        seen_nodes = int_fill_var_cuda(e_s.size(), kg.dummy_e).unsqueeze(1)
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

    def sample_action(self, db_outcomes, inv_offset=None):
        """
        Sample an action based on current policy.
        :param db_outcomes (((r_space, e_space), action_mask), action_dist):
                r_space: (Variable:batch) relation space
                e_space: (Variable:batch) target entity space
                action_mask: (Variable:batch) binary mask indicating padding actions.
                action_dist: (Variable:batch) action distribution of the current step based on set_policy
                    network parameters
        :param inv_offset: Indexes for restoring original order in a batch.
        :return next_action (next_r, next_e): Sampled next action.
        :return action_prob: Probability of the sampled action.
        """

        def apply_action_dropout_mask(action_dist, action_mask):
            if self.action_dropout_rate > 0:
                rand = torch.rand(action_dist.size())
                action_keep_mask = var_cuda(rand > self.action_dropout_rate).float()
                # There is a small chance that that action_keep_mask is accidentally set to zero.
                # When this happen, we take a random sample from the available actions.
                # sample_action_dist = action_dist * (action_keep_mask + ops.EPSILON)
                sample_action_dist = \
                    action_dist * action_keep_mask + ops.EPSILON * (1 - action_keep_mask) * action_mask
                return sample_action_dist
            else:
                return action_dist

        def sample(action_space, action_dist):
            sample_outcome = {}
            ((r_space, e_space), action_mask) = action_space
            sample_action_dist = apply_action_dropout_mask(action_dist, action_mask)
            idx = torch.multinomial(sample_action_dist, 1, replacement=True)
            next_r = ops.batch_lookup(r_space, idx)
            next_e = ops.batch_lookup(e_space, idx)
            action_prob = ops.batch_lookup(action_dist, idx)
            sample_outcome['action_sample'] = (next_r, next_e)
            sample_outcome['action_prob'] = action_prob
            return sample_outcome

        if inv_offset is not None:
            next_r_list = []
            next_e_list = []
            action_dist_list = []
            action_prob_list = []
            for action_space, action_dist in db_outcomes:
                sample_outcome = sample(action_space, action_dist)
                next_r_list.append(sample_outcome['action_sample'][0])
                next_e_list.append(sample_outcome['action_sample'][1])
                action_prob_list.append(sample_outcome['action_prob'])
                action_dist_list.append(action_dist)
            next_r = torch.cat(next_r_list, dim=0)[inv_offset]
            next_e = torch.cat(next_e_list, dim=0)[inv_offset]
            action_sample = (next_r, next_e)
            action_prob = torch.cat(action_prob_list, dim=0)[inv_offset]
            sample_outcome = {}
            sample_outcome['action_sample'] = action_sample
            sample_outcome['action_prob'] = action_prob
        else:
            sample_outcome = sample(db_outcomes[0][0], db_outcomes[0][1])

        return sample_outcome

    def predict(self, mini_batch, verbose=False):
        kg, pn = self.kg, self.mdl
        e1, e2, r = self.format_batch(mini_batch)
        beam_search_output = search.beam_search(
            pn, e1, r, e2, kg, self.num_rollout_steps, self.beam_size)
        pred_e2s = beam_search_output['pred_e2s']
        pred_e2_scores = beam_search_output['pred_e2_scores']
        if verbose:
            # print inference paths
            search_traces = beam_search_output['search_traces']
            output_beam_size = min(self.beam_size, pred_e2_scores.shape[1])
            for i in range(len(e1)):
                for j in range(output_beam_size):
                    ind = i * output_beam_size + j
                    if pred_e2s[i][j] == kg.dummy_e:
                        break
                    search_trace = []
                    for k in range(len(search_traces)):
                        search_trace.append((int(search_traces[k][0][ind]), int(search_traces[k][1][ind])))
                    print('beam {}: score = {} \n<PATH> {}'.format(
                        j, float(pred_e2_scores[i][j]), ops.format_path(search_trace, kg)))
        with torch.no_grad():
            pred_scores = zeros_var_cuda([len(e1), kg.num_entities])
            for i in range(len(e1)):
                pred_scores[i][pred_e2s[i]] = torch.exp(pred_e2_scores[i])
        return pred_scores

    def record_path_trace(self, path_trace):
        path_length = len(path_trace)
        flattened_path_trace = [x for t in path_trace for x in t]
        path_trace_mat = torch.cat(flattened_path_trace).reshape(-1, path_length)
        path_trace_mat = path_trace_mat.data.cpu().numpy()

        for i in range(path_trace_mat.shape[0]):
            path_recorder = self.path_types
            for j in range(path_trace_mat.shape[1]):
                e = path_trace_mat[i, j]
                if e not in path_recorder:
                    if j == path_trace_mat.shape[1] - 1:
                        path_recorder[e] = 1
                        self.num_path_types += 1
                    else:
                        path_recorder[e] = {}
                else:
                    if j == path_trace_mat.shape[1] - 1:
                        path_recorder[e] += 1
                path_recorder = path_recorder[e]
