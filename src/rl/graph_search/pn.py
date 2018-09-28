# -*- coding: utf-8 -*-
"""
Graph Search Policy Network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils.ops as ops
from src.utils.ops import var_cuda, zeros_var_cuda


class GraphSearchPolicy(nn.Module):
    def __init__(self, args):
        super(GraphSearchPolicy, self).__init__()
        self.model = args.model
        self.relation_only = args.relation_only

        self.history_dim = args.history_dim
        self.history_num_layers = args.history_num_layers
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        if self.relation_only:
            self.action_dim = args.relation_dim
        else:
            self.action_dim = args.entity_dim + args.relation_dim
        self.ff_dropout_rate = args.ff_dropout_rate
        self.rnn_dropout_rate = args.rnn_dropout_rate
        self.action_dropout_rate = args.action_dropout_rate

        self.xavier_initialization = args.xavier_initialization

        self.relation_only_in_path = args.relation_only_in_path
        self.path = None

        # Set policy network modules
        self.define_modules()
        self.initialize_modules()

        # Fact network modules
        self.fn = None
        self.fn_kg = None

    def transit(self, e, obs, kg, use_action_space_bucketing=True, merge_aspace_batching_outcome=False):
        """
        Compute the next action distribution based on
            (a) the current node (entity) in KG and the query relation
            (b) action history representation
        :param e: agent location (node) at step t.
        :param obs: agent observation at step t.
            e_s: source node
            q: query relation
            e_t: target node
            last_step: If set, the agent is carrying out the last step.
            last_r: label of edge traversed in the previous step
            seen_nodes: notes seen on the paths
        :param kg: Knowledge graph environment.
        :param use_action_space_bucketing: If set, the action space of different nodes are set into buckets.
        :param merge_aspace_batch_outcome: If set, merge the transition probability distribution
            generated of different action space bucket into a single batch.
        :return
            With aspace batching and without merging the outcomes:
                db_outcomes: (Dynamic Batch) (action_space, action_dist)
                    action_space: (Batch) padded possible action indices
                    action_dist: (Batch) distribution over actions.
                inv_offset: Indices to set the dynamic batching output back to the original order.
                entropy: (Batch) entropy of action distribution.
            Else:
                action_dist: (Batch) distribution over actions.
                entropy: (Batch) entropy of action distribution.
        """
        e_s, q, e_t, last_step, last_r, seen_nodes = obs

        # Representation of the current state (current node and other observations)
        Q = kg.get_relation_embeddings(q)
        H = self.path[-1][0][-1, :, :]
        if self.relation_only:
            X = torch.cat([H, Q], dim=-1)
        elif self.relation_only_in_path:
            E_s = kg.get_entity_embeddings(e_s)
            E = kg.get_entity_embeddings(e)
            X = torch.cat([E, H, E_s, Q], dim=-1)
        else:
            E = kg.get_entity_embeddings(e)
            X = torch.cat([E, H, Q], dim=-1)

        # MLP
        X = self.W1(X)
        X = F.relu(X)
        X = self.W1Dropout(X)
        X = self.W2(X)
        X2 = self.W2Dropout(X)

        def policy_nn_fun(X2, action_space):
            (r_space, e_space), action_mask = action_space
            A = self.get_action_embedding((r_space, e_space), kg)
            action_dist = F.softmax(
                torch.squeeze(A @ torch.unsqueeze(X2, 2), 2) - (1 - action_mask) * ops.HUGE_INT, dim=-1)
            # action_dist = ops.weighted_softmax(torch.squeeze(A @ torch.unsqueeze(X2, 2), 2), action_mask)
            return action_dist, ops.entropy(action_dist)

        def pad_and_cat_action_space(action_spaces, inv_offset):
            db_r_space, db_e_space, db_action_mask = [], [], []
            for (r_space, e_space), action_mask in action_spaces:
                db_r_space.append(r_space)
                db_e_space.append(e_space)
                db_action_mask.append(action_mask)
            r_space = ops.pad_and_cat(db_r_space, padding_value=kg.dummy_r)[inv_offset]
            e_space = ops.pad_and_cat(db_e_space, padding_value=kg.dummy_e)[inv_offset]
            action_mask = ops.pad_and_cat(db_action_mask, padding_value=0)[inv_offset]
            action_space = ((r_space, e_space), action_mask)
            return action_space

        if use_action_space_bucketing:
            db_outcomes = []
            entropy_list = []
            references = []
            db_action_spaces, db_references = self.get_batched_action_space(e, obs, kg)
            for action_space_b, reference_b in zip(db_action_spaces, db_references):
                X2_b = X2[reference_b, :]
                action_dist_b, entropy_b = policy_nn_fun(X2_b, action_space_b)
                references.extend(reference_b)
                db_outcomes.append((action_space_b, action_dist_b))
                entropy_list.append(entropy_b)
            inv_offset = [i for i, _ in sorted(enumerate(references), key=lambda x: x[1])]
            entropy = torch.cat(entropy_list, dim=0)[inv_offset]
            if merge_aspace_batching_outcome:
                db_action_dist = []
                for _, action_dist in db_outcomes:
                    db_action_dist.append(action_dist)
                action_space = pad_and_cat_action_space(db_action_spaces, inv_offset)
                action_dist = ops.pad_and_cat(db_action_dist, padding_value=0)[inv_offset]
                db_outcomes = [(action_space, action_dist)]
                inv_offset = None
        else:
            action_space = self.get_action_space(e, obs, kg)
            action_dist, entropy = policy_nn_fun(X2, action_space)
            db_outcomes = [(action_space, action_dist)]
            inv_offset = None

        return db_outcomes, inv_offset, entropy

    def initialize_path(self, init_action, kg):
        # [batch_size, action_dim]
        if self.relation_only_in_path:
            init_action_embedding = kg.get_relation_embeddings(init_action[0])
        else:
            init_action_embedding = self.get_action_embedding(init_action, kg)
        init_action_embedding.unsqueeze_(1)
        # [num_layers, batch_size, dim]
        init_h = zeros_var_cuda([self.history_num_layers, len(init_action_embedding), self.history_dim])
        init_c = zeros_var_cuda([self.history_num_layers, len(init_action_embedding), self.history_dim])
        self.path = [self.path_encoder(init_action_embedding, (init_h, init_c))[1]]

    def update_path(self, action, kg, offset=None):
        """
        Once an action was selected, update the action history.
        :param action (r, e): (Variable:batch) indices of the most recent action
            - r is the most recently traversed edge;
            - e is the destination entity.
        :param offset: (Variable:batch) if None, adjust path history with the given offset, used for search
        :param KG: Knowledge graph environment.
        """
        def offset_path_history(p, offset):
            for i, x in enumerate(p):
                if type(x) is tuple:
                    new_tuple = tuple([_x[:, offset, :] for _x in x])
                    p[i] = new_tuple
                else:
                    p[i] = x[offset, :]

        # update action history
        if self.relation_only_in_path:
            action_embedding = kg.get_relation_embeddings(action[0])
        else:
            action_embedding = self.get_action_embedding(action, kg)
        if offset is not None:
            offset_path_history(self.path, offset)

        self.path.append(self.path_encoder(action_embedding.unsqueeze(1), self.path[-1])[1])

    def get_batched_action_space(self, e, obs, kg, collapse_entities=False):
        e_s, q, e_t, last_step, last_r, seen_nodes = obs
        assert(len(e) == len(last_r))
        assert(len(e) == len(e_s))
        assert(len(e) == len(q))
        assert(len(e) == len(e_t))
        db_action_spaces, db_references = [], []

        if collapse_entities:
            raise NotImplementedError
        else:
            entity2bucketid = kg.entity2bucketid[e.tolist()]
            key1 = entity2bucketid[:, 0]
            key2 = entity2bucketid[:, 1]
            db_ref_dict = {}
            for i in range(len(e)):
                key = int(key1[i])
                if not key in db_ref_dict:
                    db_ref_dict[key] = []
                db_ref_dict[key].append(i)
            for key in db_ref_dict:
                action_space = kg.action_space_buckets[key]
                l_bucket_ids = db_ref_dict[key]
                db_references.append(l_bucket_ids)
                g_bucket_ids = key2[l_bucket_ids].tolist()
                r_space_b = action_space[0][0][g_bucket_ids]
                e_space_b = action_space[0][1][g_bucket_ids]
                action_mask_b = action_space[1][g_bucket_ids]
                e_b = e[l_bucket_ids]
                last_r_b = last_r[l_bucket_ids]
                e_s_b = e_s[l_bucket_ids]
                q_b = q[l_bucket_ids]
                e_t_b = e_t[l_bucket_ids]
                seen_nodes_b = seen_nodes[l_bucket_ids]
                obs_b = [e_s_b, q_b, e_t_b, last_step, last_r_b, seen_nodes_b]
                action_space_b = ((r_space_b, e_space_b), action_mask_b)
                action_space_b = self.apply_action_masks(action_space_b, e_b, obs_b, kg)
                db_action_spaces.append(action_space_b)

        return db_action_spaces, db_references

    def get_action_space(self, e, obs, kg):
        r_space, e_space = kg.action_space[0][0][e], kg.action_space[0][1][e]
        action_mask = kg.action_space[1][e]
        action_space = ((r_space, e_space), action_mask)
        return self.apply_action_masks(action_space, e, obs, kg)

    def apply_action_masks(self, action_space, e, obs, kg):
        (r_space, e_space), action_mask = action_space
        e_s, q, e_t, last_step, last_r, seen_nodes = obs

        # Prevent the agent from selecting the ground truth edge
        ground_truth_edge_mask = self.get_ground_truth_edge_mask(e, r_space, e_space, e_s, q, e_t, kg)
        action_mask -= ground_truth_edge_mask
        self.validate_action_mask(action_mask)

        # Mask out false negatives in the final step
        if last_step:
            false_negative_mask = self.get_false_negative_mask(e_space, e_s, q, e_t, kg)
            action_mask *= (1 - false_negative_mask)
            self.validate_action_mask(action_mask)
            if self.training and self.model.endswith('.rsm'):
                fuzzy_false_negative_mask = self.get_fuzzy_false_negative_mask(e_space, e_s, q, e_t)
                action_mask *= (1 - fuzzy_false_negative_mask)

        # Prevent the agent from stopping in the middle of a path
        # stop_mask = (last_r == NO_OP_RELATION_ID).unsqueeze(1).float()
        # action_mask = (1 - stop_mask) * action_mask + stop_mask * (r_space == NO_OP_RELATION_ID).float()
        # Prevent loops
        # Note: avoid duplicate removal of self-loops
        # seen_nodes_b = seen_nodes[l_bucket_ids]
        # loop_mask_b = (((seen_nodes_b.unsqueeze(1) == e_space.unsqueeze(2)).sum(2) > 0) *
        #      (r_space != NO_OP_RELATION_ID)).float()
        # action_mask *= (1 - loop_mask_b)
        return (r_space, e_space), action_mask

    def get_ground_truth_edge_mask(self, e, r_space, e_space, e_s, q, e_t, kg):
        ground_truth_edge_mask = \
            ((e == e_s).unsqueeze(1) * (r_space == q.unsqueeze(1)) * (e_space == e_t.unsqueeze(1)))
        inv_q = kg.get_inv_relation_id(q)
        inv_ground_truth_edge_mask = \
            ((e == e_t).unsqueeze(1) * (r_space == inv_q.unsqueeze(1)) * (e_space == e_s.unsqueeze(1)))
        return ((ground_truth_edge_mask + inv_ground_truth_edge_mask) * (e_s.unsqueeze(1) != kg.dummy_e)).float()

    def get_answer_mask(self, e_space, e_s, q, kg):
        if kg.args.mask_test_false_negatives:
            answer_vectors = kg.all_object_vectors
        else:
            answer_vectors = kg.train_object_vectors
        answer_masks = []
        for i in range(len(e_space)):
            _e_s, _q = int(e_s[i]), int(q[i])
            if not _e_s in answer_vectors or not _q in answer_vectors[_e_s]:
                answer_vector = var_cuda(torch.LongTensor([[kg.num_entities]]))
            else:
                answer_vector = answer_vectors[_e_s][_q]
            answer_mask = torch.sum(e_space[i].unsqueeze(0) == answer_vector, dim=0).long()
            answer_masks.append(answer_mask)
        answer_mask = torch.cat(answer_masks).view(len(e_space), -1)
        return answer_mask

    def get_false_negative_mask(self, e_space, e_s, q, e_t, kg):
        answer_mask = self.get_answer_mask(e_space, e_s, q, kg)
        # This is a trick applied during training where we convert a multi-answer predction problem into several
        # single-answer prediction problems. By masking out the other answers in the training set, we are forcing
        # the agent to walk towards a particular answer.
        # This trick does not affect inference on the test set: at inference time the ground truth answer will not 
        # appear in the answer mask. This can be checked by uncommenting the following assertion statement. 
        # Note that the assertion statement can trigger in the last batch if you're using a batch_size > 1 since
        # we append dummy examples to the last batch to make it the required batch size.
        # The assertion statement will also trigger in the dev set inference of NELL-995 since we randomly 
        # sampled the dev set from the training data.
        # assert(float((answer_mask * (e_space == e_t.unsqueeze(1)).long()).sum()) == 0)
        false_negative_mask = (answer_mask * (e_space != e_t.unsqueeze(1)).long()).float()
        return false_negative_mask

    def get_fuzzy_false_negative_mask(self, e_space, e_s, q, e_t):
        e_s = ops.tile_along_beam(e_s, e_space.size(1))
        q = ops.tile_along_beam(q, e_space.size(1))
        pred_e2 = e_space.view(-1)
        scores = []
        b_s = 16000
        for i in range(0, len(e_s), b_s):
            score = self.fn.forward_fact(e_s[i:i+b_s], q[i:i+b_s], pred_e2[i:i+b_s], self.fn_kg)
            scores.append(score.detach())
        scores = torch.cat(scores).view(len(e_t), -1)
        # scores = self.fn.forward_fact(e_s, q, pred_e2, self.fn_kg)
        # scores = scores.view(len(e_t), -1)
        fuzzy_false_negative_mask = scores * (e_space != e_t.unsqueeze(1)).float()
        return fuzzy_false_negative_mask

    def validate_action_mask(self, action_mask):
        action_mask_min = action_mask.min()
        action_mask_max = action_mask.max()
        assert (action_mask_min == 0 or action_mask_min == 1)
        assert (action_mask_max == 0 or action_mask_max == 1)

    def get_action_embedding(self, action, kg):
        """
        Return (batch) action embedding which is the concatenation of the embeddings of
        the traversed edge and the target node.

        :param action (r, e):
            (Variable:batch) indices of the most recent action
                - r is the most recently traversed edge
                - e is the destination entity.
        :param kg: Knowledge graph enviroment.
        """
        r, e = action
        relation_embedding = kg.get_relation_embeddings(r)
        if self.relation_only:
            action_embedding = relation_embedding
        else:
            entity_embedding = kg.get_entity_embeddings(e)
            action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)
        return action_embedding

    def define_modules(self):
        if self.relation_only:
            input_dim = self.history_dim + self.relation_dim
        elif self.relation_only_in_path:
            input_dim = self.history_dim + self.entity_dim * 2 + self.relation_dim
        else:
            input_dim = self.history_dim + self.entity_dim + self.relation_dim
        self.W1 = nn.Linear(input_dim, self.action_dim)
        self.W2 = nn.Linear(self.action_dim, self.action_dim)
        self.W1Dropout = nn.Dropout(p=self.ff_dropout_rate)
        self.W2Dropout = nn.Dropout(p=self.ff_dropout_rate)
        if self.relation_only_in_path:
            self.path_encoder = nn.LSTM(input_size=self.relation_dim,
                                        hidden_size=self.history_dim,
                                        num_layers=self.history_num_layers,
                                        batch_first=True)
        else:
            self.path_encoder = nn.LSTM(input_size=self.action_dim,
                                        hidden_size=self.history_dim,
                                        num_layers=self.history_num_layers,
                                        batch_first=True)

    def initialize_modules(self):
        if self.xavier_initialization:
            nn.init.xavier_uniform_(self.W1.weight)
            nn.init.xavier_uniform_(self.W2.weight)
            for name, param in self.path_encoder.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
