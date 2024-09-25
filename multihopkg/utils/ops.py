"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Customized operators and utility functions.
"""

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

EPSILON = float(np.finfo(float).eps)
HUGE_INT = 1e31


def batch_lookup(M, idx, vector_output=True):
    """
    Perform batch lookup on matrix M using indices idx.
    :param M: (Variable) [batch_size, seq_len] Each row of M is an independent population.
    :param idx: (Variable) [batch_size, sample_size] Each row of idx is a list of sample indices.
    :param vector_output: If set, return a 1-D vector when sample size is 1.
    :return samples: [batch_size, sample_size] samples[i, j] = M[idx[i, j]]
    """
    batch_size, w = M.size()
    batch_size2, sample_size = idx.size()
    assert(batch_size == batch_size2)

    if sample_size == 1 and vector_output:
        samples = torch.gather(M, 1, idx).view(-1)
    else:
        samples = torch.gather(M, 1, idx)
    return samples


def convert_to_dist(x):
    x += EPSILON
    return x / x.sum(1, keepdim=True)


def detach_module(mdl):
    for param in mdl.parameters():
        param.requires_grad = False


def entropy(p):
    return torch.sum(-p * safe_log(p), 1)


def weighted_softmax(v, w, dim=-1):
    exp_v = torch.exp(v)
    weighted_exp_v = w * exp_v
    return weighted_exp_v / torch.sum(weighted_exp_v, dim, keepdim=True)


def format_triple(triple, kg):
    e1, e2, r = triple
    rel = kg.id2relation[r] if r != kg.self_edge else '<null>'
    if not rel.endswith('_inv'):
        return '{} -{}-> {}'.format(
            kg.id2entity[e1], rel, kg.id2entity[e2])
    else:
        return '{} <-{}- {}'.format(
            kg.id2entity[e1], rel, kg.id2entity[e2])


def format_path(path_trace, kg):
    def get_most_recent_relation(j):
        relation_id = int(path_trace[j][0])
        if relation_id == kg.self_edge:
            return '<null>'
        else:
            return kg.id2relation[relation_id]

    def get_most_recent_entity(j):
        return kg.id2entity[int(path_trace[j][1])]

    path_str = get_most_recent_entity(0)
    for j in range(1, len(path_trace)):
        rel = get_most_recent_relation(j)
        if not rel.endswith('_inv'):
            path_str += ' -{}-> '.format(rel)
        else:
            path_str += ' <-{}- '.format(rel[:-4])
        path_str += get_most_recent_entity(j)
    return path_str


def format_rule(rule, kg):
    rule_str = ''
    for j in range(len(rule)):
        relation_id = int(rule[j])
        rel = kg.id2relation[relation_id]
        if not rel.endswith('_inv'):
            rule_str += '-{}-> '.format(rel)
        else:
            rule_str += '<-{}-'.format(rel)
    return rule_str


def ones_var_cuda(s, requires_grad=False):
    return Variable(torch.ones(s), requires_grad=requires_grad).cuda()


def zeros_var_cuda(s, requires_grad=False):
    return Variable(torch.zeros(s), requires_grad=requires_grad).cuda()


def int_fill_var_cuda(s, value, requires_grad=False):
    return int_var_cuda((torch.zeros(s) + value), requires_grad=requires_grad)


def int_var_cuda(x, requires_grad=False):
    return Variable(x, requires_grad=requires_grad).long().cuda()


def var_cuda(x, requires_grad=False):
    return Variable(x, requires_grad=requires_grad).cuda()


def var_to_numpy(x):
    return x.data.cpu().numpy()


def pad_and_cat(a, padding_value, padding_dim=1):
    max_dim_size = max([x.size()[padding_dim] for x in a])
    padded_a = []
    for x in a:
        if x.size()[padding_dim] < max_dim_size:
            res_len = max_dim_size - x.size()[1]
            pad = nn.ConstantPad1d((0, res_len), padding_value)
            padded_a.append(pad(x))
        else:
            padded_a.append(x)
    return torch.cat(padded_a, dim=0)


def rearrange_vector_list(l, offset):
    for i, v in enumerate(l):
        l[i] = v[offset]

def safe_log(x):
    return torch.log(x + EPSILON)


def tile_along_beam(v, beam_size, dim=0):
    """
    Tile a tensor along a specified dimension for the specified beam size.
    :param v: Input tensor.
    :param beam_size: Beam size.
    """
    if dim == -1:
        dim = len(v.size()) - 1
    v = v.unsqueeze(dim + 1)
    v = torch.cat([v] * beam_size, dim=dim+1)
    new_size = []
    for i, d in enumerate(v.size()):
        if i == dim + 1:
            new_size[-1] *= d
        else:
            new_size.append(d)
    return v.view(new_size)


# Flatten and pack nested lists using recursion
def flatten(l):
    flatten_l = []
    for c in l:
        if type(c) is list or type(c) is tuple:
            flatten_l.extend(flatten(c))
        else:
            flatten_l.append(c)
    return flatten_l


def pack(l, a):
    """
    Pack a flattened list l into the structure of the nested list a.
    """
    nested_l = []
    for c in a:
        if type(c) is not list:
            nested_l.insert(l[0], 0)
            l.pop(0)


def unique_max(unique_x, x, values, marker_2D=None):
    unique_interval = 100
    unique_values, unique_indices = [], []
    # prevent memory explotion during decoding
    for i in range(0, len(unique_x), unique_interval):
        unique_x_b = unique_x[i:i+unique_interval]
        marker_2D = (unique_x_b.unsqueeze(1) == x.unsqueeze(0)).float()
        values_2D = marker_2D * values.unsqueeze(0) - (1 - marker_2D) * HUGE_INT
        unique_values_b, unique_idx_b = values_2D.max(dim=1)
        unique_values.append(unique_values_b)
        unique_indices.append(unique_idx_b)
    unique_values = torch.cat(unique_values)
    unique_idx = torch.cat(unique_indices)
    return unique_values, unique_idx


if __name__ == '__main__':
    a = torch.randn(2)
    print(a)
    print(tile_along_beam(a, 4))
    print('--------------------------')
    b = torch.randn(2, 3)
    print(b)
    c = tile_along_beam(b, 4)
    print(c)
    print('--------------------------')
    print(c.view(2, -1))
