"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Policy gradient with reward shaping.
"""

from tqdm import tqdm

import torch

from src.emb.fact_network import get_conve_nn_state_dict, get_conve_kg_state_dict, \
    get_complex_kg_state_dict, get_distmult_kg_state_dict
from src.rl.graph_search.pg import PolicyGradient
import src.utils.ops as ops
from src.utils.ops import zeros_var_cuda


class RewardShapingPolicyGradient(PolicyGradient):
    def __init__(self, args, kg, pn, fn_kg, fn, fn_secondary_kg=None):
        super(RewardShapingPolicyGradient, self).__init__(args, kg, pn)
        self.reward_shaping_threshold = args.reward_shaping_threshold

        # Fact network modules
        self.fn_kg = fn_kg
        self.fn = fn
        self.fn_secondary_kg = fn_secondary_kg
        self.mu = args.mu

        fn_model = self.fn_model
        if fn_model in ['conve']:
            fn_state_dict = torch.load(args.conve_state_dict_path)
            fn_nn_state_dict = get_conve_nn_state_dict(fn_state_dict)
            fn_kg_state_dict = get_conve_kg_state_dict(fn_state_dict)
            self.fn.load_state_dict(fn_nn_state_dict)
        elif fn_model == 'distmult':
            fn_state_dict = torch.load(args.distmult_state_dict_path)
            fn_kg_state_dict = get_distmult_kg_state_dict(fn_state_dict)
        elif fn_model == 'complex':
            fn_state_dict = torch.load(args.complex_state_dict_path)
            fn_kg_state_dict = get_complex_kg_state_dict(fn_state_dict)
        elif fn_model == 'hypere':
            fn_state_dict = torch.load(args.conve_state_dict_path)
            fn_kg_state_dict = get_conve_kg_state_dict(fn_state_dict)
        else:
            raise NotImplementedError
        self.fn_kg.load_state_dict(fn_kg_state_dict)
        if fn_model == 'hypere':
            complex_state_dict = torch.load(args.complex_state_dict_path)
            complex_kg_state_dict = get_complex_kg_state_dict(complex_state_dict)
            self.fn_secondary_kg.load_state_dict(complex_kg_state_dict)

        self.fn.eval()
        self.fn_kg.eval()
        ops.detach_module(self.fn)
        ops.detach_module(self.fn_kg)
        if fn_model == 'hypere':
            self.fn_secondary_kg.eval()
            ops.detach_module(self.fn_secondary_kg)

    def reward_fun(self, e1, r, e2, pred_e2):
        if self.model.endswith('.rso'):
            oracle_reward = forward_fact_oracle(e1, r, pred_e2, self.kg)
            return oracle_reward
        else:
            if self.fn_secondary_kg:
                real_reward = self.fn.forward_fact(e1, r, pred_e2, self.fn_kg, [self.fn_secondary_kg]).squeeze(1)
            else:
                real_reward = self.fn.forward_fact(e1, r, pred_e2, self.fn_kg).squeeze(1)
            real_reward_mask = (real_reward > self.reward_shaping_threshold).float()
            real_reward *= real_reward_mask
            if self.model.endswith('rsc'):
                return real_reward
            else:
                binary_reward = (pred_e2 == e2).float()
                return binary_reward + self.mu * (1 - binary_reward) * real_reward

    def test_fn(self, examples):
        fn_kg, fn = self.fn_kg, self.fn
        pred_scores = []
        for example_id in tqdm(range(0, len(examples), self.batch_size)):
            mini_batch = examples[example_id:example_id + self.batch_size]
            mini_batch_size = len(mini_batch)
            if len(mini_batch) < self.batch_size:
                self.make_full_batch(mini_batch, self.batch_size)
            e1, e2, r = self.format_batch(mini_batch)
            if self.fn_secondary_kg:
                pred_score = fn.forward_fact(e1, r, e2, fn_kg, [self.fn_secondary_kg])
            else:
                pred_score = fn.forward_fact(e1, r, e2, fn_kg)
            pred_scores.append(pred_score[:mini_batch_size])
        return torch.cat(pred_scores)

    @property
    def fn_model(self):
        return self.model.split('.')[2]

def forward_fact_oracle(e1, r, e2, kg):
    oracle = zeros_var_cuda([len(e1), kg.num_entities]).cuda()
    for i in range(len(e1)):
        _e1, _r = int(e1[i]), int(r[i])
        if _e1 in kg.all_object_vectors and _r in kg.all_object_vectors[_e1]:
            answer_vector = kg.all_object_vectors[_e1][_r]
            oracle[i][answer_vector] = 1
        else:
            raise ValueError('Query answer not found')
    oracle_e2 = ops.batch_lookup(oracle, e2.unsqueeze(1))
    return oracle_e2
