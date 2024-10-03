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

# From transformers import general tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizer

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
from multihopkg.run_configs import alpha
from multihopkg.rl.graph_search.pg import PolicyGradient
from multihopkg.rl.graph_search.pn import GraphSearchPolicy
from multihopkg.rl.graph_search.rs_pg import RewardShapingPolicyGradient
from multihopkg.utils.ops import flatten
from multihopkg.logging import setup_logger
from multihopkg.utils.setup import set_seeds
from multihopkg.models.construction import (
    construct_env_model,
    construct_navagent,
    construct_embedding_model,
)
from typing import Any, Dict, Tuple
import pdb





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


def train(lf):
    train_path = data_utils.get_train_path(args)
    dev_path = os.path.join(args.data_dir, "dev.triples")
    entity_index_path = os.path.join(args.data_dir, "entity2id.txt")
    relation_index_path = os.path.join(args.data_dir, "relation2id.txt")
    train_data = data_utils.load_triples(
        train_path,
        entity_index_path,
        relation_index_path,
        group_examples_by_query=args.group_examples_by_query,
        add_reverse_relations=args.add_reversed_training_edges,
    )
    # NELL is a dataset
    if "NELL" in args.data_dir:
        adj_list_path = os.path.join(args.data_dir, "adj_list.pkl")
        seen_entities = data_utils.load_seen_entities(adj_list_path, entity_index_path)
    else:
        seen_entities = set()
    dev_data = data_utils.load_triples(
        dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities
    )
    if args.checkpoint_path is not None:
        lf.load_checkpoint(args.checkpoint_path)
    lf.run_train(train_data, dev_data)


def inference(lf):
    lf.batch_size = args.dev_batch_size
    lf.eval()
    if args.model == "hypere":
        conve_kg_state_dict = get_conve_kg_state_dict(
            torch.load(args.conve_state_dict_path)
        )
        lf.kg.load_state_dict(conve_kg_state_dict)
        secondary_kg_state_dict = get_complex_kg_state_dict(
            torch.load(args.complex_state_dict_path)
        )
        lf.secondary_kg.load_state_dict(secondary_kg_state_dict)
    elif args.model == "triplee":
        conve_kg_state_dict = get_conve_kg_state_dict(
            torch.load(args.conve_state_dict_path)
        )
        lf.kg.load_state_dict(conve_kg_state_dict)
        complex_kg_state_dict = get_complex_kg_state_dict(
            torch.load(args.complex_state_dict_path)
        )
        lf.secondary_kg.load_state_dict(complex_kg_state_dict)
        distmult_kg_state_dict = get_distmult_kg_state_dict(
            torch.load(args.distmult_state_dict_path)
        )
        lf.tertiary_kg.load_state_dict(distmult_kg_state_dict)
    else:
        lf.load_checkpoint(get_checkpoint_path(args))
    entity_index_path = os.path.join(args.data_dir, "entity2id.txt")
    relation_index_path = os.path.join(args.data_dir, "relation2id.txt")
    if "NELL" in args.data_dir:
        adj_list_path = os.path.join(args.data_dir, "adj_list.pkl")
        seen_entities = data_utils.load_seen_entities(adj_list_path, entity_index_path)
    else:
        seen_entities = set()

    eval_metrics = {"dev": {}, "test": {}}

    if args.compute_map:
        relation_sets = [
            "concept:athletehomestadium",
            "concept:athleteplaysforteam",
            "concept:athleteplaysinleague",
            "concept:athleteplayssport",
            "concept:organizationheadquarteredincity",
            "concept:organizationhiredperson",
            "concept:personborninlocation",
            "concept:teamplayssport",
            "concept:worksfor",
        ]
        mps = []
        for r in relation_sets:
            print("* relation: {}".format(r))
            test_path = os.path.join(args.data_dir, "tasks", r, "test.pairs")
            test_data, labels = data_utils.load_triples_with_label(
                test_path,
                r,
                entity_index_path,
                relation_index_path,
                seen_entities=seen_entities,
            )
            pred_scores = lf.forward(test_data, verbose=False)
            mp = src.eval.link_MAP(
                test_data, pred_scores, labels, lf.kg.all_objects, verbose=True
            )
            mps.append(mp)
        map_ = np.mean(mps)
        print("Overall MAP = {}".format(map_))
        eval_metrics["test"]["avg_map"] = map
    elif args.eval_by_relation_type:
        dev_path = os.path.join(args.data_dir, "dev.triples")
        dev_data = data_utils.load_triples(
            dev_path,
            entity_index_path,
            relation_index_path,
            seen_entities=seen_entities,
        )
        pred_scores = lf.forward(dev_data, verbose=False)
        to_m_rels, to_1_rels, _ = data_utils.get_relations_by_type(
            args.data_dir, relation_index_path
        )
        relation_by_types = (to_m_rels, to_1_rels)
        print("Dev set evaluation by relation type (partial graph)")
        src.eval.hits_and_ranks_by_relation_type(
            dev_data, pred_scores, lf.kg.dev_objects, relation_by_types, verbose=True
        )
        print("Dev set evaluation by relation type (full graph)")
        src.eval.hits_and_ranks_by_relation_type(
            dev_data, pred_scores, lf.kg.all_objects, relation_by_types, verbose=True
        )
    elif args.eval_by_seen_queries:
        dev_path = os.path.join(args.data_dir, "dev.triples")
        dev_data = data_utils.load_triples(
            dev_path,
            entity_index_path,
            relation_index_path,
            seen_entities=seen_entities,
        )
        pred_scores = lf.forward(dev_data, verbose=False)
        seen_queries = data_utils.get_seen_queries(
            args.data_dir, entity_index_path, relation_index_path
        )
        print("Dev set evaluation by seen queries (partial graph)")
        src.eval.hits_and_ranks_by_seen_queries(
            dev_data, pred_scores, lf.kg.dev_objects, seen_queries, verbose=True
        )
        print("Dev set evaluation by seen queries (full graph)")
        src.eval.hits_and_ranks_by_seen_queries(
            dev_data, pred_scores, lf.kg.all_objects, seen_queries, verbose=True
        )
    else:
        dev_path = os.path.join(args.data_dir, "dev.triples")
        test_path = os.path.join(args.data_dir, "test.triples")
        dev_data = data_utils.load_triples(
            dev_path,
            entity_index_path,
            relation_index_path,
            seen_entities=seen_entities,
            verbose=False,
        )
        test_data = data_utils.load_triples(
            test_path,
            entity_index_path,
            relation_index_path,
            seen_entities=seen_entities,
            verbose=False,
        )
        print("Dev set performance:")
        pred_scores = lf.forward(dev_data, verbose=args.save_beam_search_paths)
        dev_metrics = src.eval.hits_and_ranks(
            dev_data, pred_scores, lf.kg.dev_objects, verbose=True
        )
        eval_metrics["dev"] = {}
        eval_metrics["dev"]["hits_at_1"] = dev_metrics[0]
        eval_metrics["dev"]["hits_at_3"] = dev_metrics[1]
        eval_metrics["dev"]["hits_at_5"] = dev_metrics[2]
        eval_metrics["dev"]["hits_at_10"] = dev_metrics[3]
        eval_metrics["dev"]["mrr"] = dev_metrics[4]
        src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.all_objects, verbose=True)
        print("Test set performance:")
        pred_scores = lf.forward(test_data, verbose=False)
        test_metrics = src.eval.hits_and_ranks(
            test_data, pred_scores, lf.kg.all_objects, verbose=True
        )
        eval_metrics["test"]["hits_at_1"] = test_metrics[0]
        eval_metrics["test"]["hits_at_3"] = test_metrics[1]
        eval_metrics["test"]["hits_at_5"] = test_metrics[2]
        eval_metrics["test"]["hits_at_10"] = test_metrics[3]
        eval_metrics["test"]["mrr"] = test_metrics[4]

    return eval_metrics


def load_configs(config_path):
    with open(config_path) as f:
        print("loading configuration file {}".format(config_path))
        for line in f:
            if not "=" in line:
                continue
            arg_name, arg_value = line.strip().split("=")
            if arg_value.startswith('"') and arg_value.endswith('"'):
                arg_value = arg_value[1:-1]
            if hasattr(args, arg_name):
                print("{} = {}".format(arg_name, arg_value))
                arg_value2 = getattr(args, arg_name)
                if type(arg_value2) is str:
                    setattr(args, arg_name, arg_value)
                elif type(arg_value2) is bool:
                    if arg_value == "True":
                        setattr(args, arg_name, True)
                    elif arg_value == "False":
                        setattr(args, arg_name, False)
                    else:
                        raise ValueError(
                            "Unrecognized boolean value description: {}".format(
                                arg_value
                            )
                        )
                elif type(arg_value2) is int:
                    setattr(args, arg_name, int(arg_value))
                elif type(arg_value2) is float:
                    setattr(args, arg_name, float(arg_value))
                else:
                    raise ValueError(
                        "Unrecognized attribute type: {}: {}".format(
                            arg_name, type(arg_value2)
                        )
                    )
            else:
                raise ValueError("Unrecognized argument: {}".format(arg_name))
    return args


def initial_setup():
    args = alpha.get_args()
    torch.cuda.set_device(args.gpu)
    set_seeds(args.seed)
    logger = setup_logger("__MAIN__")

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    return args, tokenizer, logger


def main():
    # By default we run the config
    # Process data will determine by itself if there is any data to process
    args, tokenizer, logger = initial_setup()

    logger.info("Loading data") 
    process_data(args.QAtriplets_raw_dir, args.QAtriplets_cache_dir, tokenizer)

    # TODO: Maybe re-enable this logic later
    # with torch.set_grad_enabled(args.train or args.search_random_seed or args.grid_search):
    logger.info("Setting up seeds, logger, tokenizer")
    args, tokenizer, logger = initial_setup()

    # TODO: Muybe ? (They use it themselves)
    # initialize_model_directory(args, args.seed)

    # Setting up the models
    logger.info(":: (0/3) Setting up models..")
    embedding_model = construct_embedding_model(args)
    logger.info(":: (1/3) Loaded embedding model")
    env = construct_env_model(args)
    logger.info(":: (2/3) Loaded environment module")
    nav_agent = construct_navagent(args)  # Boi to be trained
    logger.info(":: (3/3) Loaded navigation agent")

    logger.info(":: Training the model")
    train(nav_agent)

    # TODO: Evaluation of the model
    # metrics = inference(lf)


if __name__ == "__main__":
    main()
