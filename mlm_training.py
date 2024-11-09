#!/usr/bin/env python3

"""
 Copyright (c) 2018, salesforce.com, inc.
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Experiment Portal.
"""
import argparse
import json
import logging
import os
from typing import List, Tuple

import pandas as pd
import torch
from torch import nn
from rich import traceback
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertModel, PreTrainedTokenizer, BartConfig

import multihopkg.data_utils as data_utils
from multihopkg.knowledge_graph import ITLKnowledgeGraph
from multihopkg.logging import setup_logger
from multihopkg.rl.graph_search.cpg import ContinuousPolicyGradient
from multihopkg.rl.graph_search.pn import ITLGraphEnvironment
from multihopkg.run_configs import alpha
from multihopkg.utils.setup import set_seeds
from multihopkg.vector_search import ANN_IndexMan
from multihopkg.environments import Observation
from multihopkg.language_models import HunchLLM, collate_token_ids_batch
import pdb

traceback.install()


def initialize_model_directory(args, random_seed=None):
    # add model parameter info to model directory
    # TODO: We might2ant our implementation of something like this later
    raise NotImplementedError

def initial_setup() -> Tuple[argparse.Namespace, PreTrainedTokenizer, logging.Logger]:
    global logger
    args = alpha.get_args()
    torch.cuda.set_device(args.gpu)
    set_seeds(args.seed)
    logger = setup_logger("__MAIN__")

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    assert isinstance(args, argparse.Namespace)

    return args, tokenizer, logger

def prep_questions(questions: List[torch.Tensor], model: BertModel):
    embedded_questions = model(questions)
    return embedded_questions


def batch_loop(
    env: ITLGraphEnvironment,
    mini_batch: pd.DataFrame,  # Perhaps change this ?
    nav_agent: ContinuousPolicyGradient,
    hunch_llm: nn.Module,
    steps_in_episode: int,
) -> torch.Tensor:

    ########################################
    # Start the batch loop with zero grad
    ########################################
    nav_agent.zero_grad()

    # Deconstruct the batch
    questions = mini_batch['question'].tolist()
    answers = mini_batch['answer'].tolist()
    question_embeddings = env.get_llm_embeddings(questions)
    answer_ids_padded_tensor = collate_token_ids_batch(answers).to(torch.int32)

    log_probs, rewards = rollout(
        steps_in_episode,
        nav_agent,
        hunch_llm,
        env,
        question_embeddings,
        answer_ids_padded_tensor,
    )

    ########################################
    # Calculate Reinforce Objective
    ########################################
    # Compute policy gradient
    num_steps = len(log_probs)
    rewards_t = torch.stack(rewards).sum(dim=1)
    log_probs_t = torch.stack(log_probs).sum(dim=1)

    pg_loss = -1*rewards_t * log_probs_t

    return pg_loss


def train_multihopkg(
    batch_size: int,
    epochs: int,
    nav_agent: ContinuousPolicyGradient,
    hunch_llm: nn.Module,
    learning_rate: float,
    steps_in_episode: int,
    env: ITLGraphEnvironment,
    start_epoch: int,
    train_data: pd.DataFrame,
):
    # TODO: Get the rollout working

    # Print Model Parameters + Perhaps some more information
    print(
        "--------------------------\n"
    "Model Parameters\n"
    "--------------------------"
    )
    for name, param in nav_agent.named_parameters():
        print(name, param.numel(), "requires_grad={}".format(param.requires_grad))

    # Just use Adam Optimizer by default
    optimizer = torch.optim.Adam(  # type: ignore
        filter(lambda p: p.requires_grad, nav_agent.parameters()), lr=learning_rate
    )

    for epoch_id in range(start_epoch, epochs):
        logger.info("Epoch {}".format(epoch_id))
        # TODO: Perhaps evaluate the epochs?

        # Set in training mode
        nav_agent.train()

        # TOREM: Perhapas no need for this shuffle.
        batch_rewards = []
        entropies = []

        # TODO: Understand if this is actually necessary here
        # if self.run_analysis:
        #     rewards = None
        #     fns = None

        ##############################
        # Batch Iteration Starts Here.
        ##############################
        # TODO: update the parameters.
        for sample_offset_idx in tqdm(range(0, len(train_data), batch_size)):
            mini_batch = train_data[sample_offset_idx : sample_offset_idx + batch_size]
            assert isinstance(mini_batch, pd.DataFrame) # For the lsp to give me a break
            optimizer.zero_grad()
            reinforce_terms = batch_loop(
                 env, mini_batch, nav_agent, hunch_llm, steps_in_episode
            )
            reinforce_terms_mean = reinforce_terms.mean()
            batch_rewards.append(reinforce_terms_mean.item())
            reinforce_terms_mean.backward()


            optimizer.step()

            # TODO: Do something with the mini batch
        # TODO: Check on the metrics:
        # Check training statistics
        # stdout_msg = 'Epoch {}: average training loss = {}'.format(epoch_id, np.mean(batch_losses))
        # if entropies:
        #     stdout_msg += ' entropy = {}'.format(np.mean(entropies))
        # print(stdout_msg)
        # self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id)
        # if self.run_analysis:
        #     print('* Analysis: # path types seen = {}'.format(self.num_path_types))
        #     num_hits = float(rewards.sum())
        #     hit_ratio = num_hits / len(rewards)
        #     print('* Analysis: # hits = {} ({})'.format(num_hits, hit_ratio))
        #     num_fns = float(fns.sum())
        #     fn_ratio = num_fns / len(fns)
        #     print('* Analysis: false negative ratio = {}'.format(fn_ratio))
        #
        # # Check dev set performance
        # if self.run_analysis or (epoch_id > 0 and epoch_id % self.num_peek_epochs == 0):
        #     self.eval()
        #     self.batch_size = self.dev_batch_size
        #     dev_scores = self.forward(dev_data, verbose=False)
        #     print('Dev set performance: (correct evaluation)')
        #     _, _, _, _, mrr = src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.dev_objects, verbose=True)
        #     metrics = mrr
        #     print('Dev set performance: (include test set labels)')
        #     src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.all_objects, verbose=True)
        #     # Action dropout anneaking
        #     if self.model.startswith('point'):
        #         eta = self.action_dropout_anneal_interval
        #         if len(dev_metrics_history) > eta and metrics < min(dev_metrics_history[-eta:]):
        #             old_action_dropout_rate = self.action_dropout_rate
        #             self.action_dropout_rate *= self.action_dropout_anneal_factor
        #             print('Decreasing action dropout rate: {} -> {}'.format(
        #                 old_action_dropout_rate, self.action_dropout_rate))
        #     # Save checkpoint
        #     if metrics > best_dev_metrics:
        #         self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id, is_best=True)
        #         best_dev_metrics = metrics
        #         with open(os.path.join(self.model_dir, 'best_dev_iteration.dat'), 'w') as o_f:
        #             o_f.write('{}'.format(epoch_id))
        #     else:
        #         # Early stopping
        #         if epoch_id >= self.num_wait_epochs and metrics < np.mean(dev_metrics_history[-self.num_wait_epochs:]):
        #             break
        #     dev_metrics_history.append(metrics)
        #     if self.run_analysis:
        #         num_path_types_file = os.path.join(self.model_dir, 'num_path_types.dat')
        #         dev_metrics_file = os.path.join(self.model_dir, 'dev_metrics.dat')
        #         hit_ratio_file = os.path.join(self.model_dir, 'hit_ratio.dat')
        #         fn_ratio_file = os.path.join(self.model_dir, 'fn_ratio.dat')
        #         if epoch_id == 0:
        #             with open(num_path_types_file, 'w') as o_f:
        #                 o_f.write('{}\n'.format(self.num_path_types))
        #             with open(dev_metrics_file, 'w') as o_f:
        #                 o_f.write('{}\n'.format(metrics))
        #             with open(hit_ratio_file, 'w') as o_f:
        #                 o_f.write('{}\n'.format(hit_ratio))
        #             with open(fn_ratio_file, 'w') as o_f:
        #                 o_f.write('{}\n'.format(fn_ratio))
        #         else:
        #             with open(num_path_types_file, 'a') as o_f:
        #                 o_f.write('{}\n'.format(self.num_path_types))
        #             with open(dev_metrics_file, 'a') as o_f:
        #                 o_f.write('{}\n'.format(metrics))
        #             with open(hit_ratio_file, 'a') as o_f:
        #                 o_f.write('{}\n'.format(hit_ratio))
        #             with open(fn_ratio_file, 'a') as o_f:
        #                 o_f.write('{}\n'.format(fn_ratio))
        #
        #


def initialize_path(questions: torch.Tensor):
    # Questions must be turned into queries
    raise NotImplementedError


def calculate_reward(hunch_llm: nn.Module, obtained_state: torch.Tensor, answers_ids: torch.Tensor) -> torch.Tensor:
    """
    Will take the answers and give an idea of how close we were.
    This will of course require us to have a language model that will start giving us the  answer.
    """
    batch_size = answers_ids.size(0)
    seq_max_len = answers_ids.size(1)

    # From the obtained_state we will try to find an answer
    answers_inf_softmax = hunch_llm(obtained_state, answers_ids)
    # Get indices of the max value of the final output
    answers_inf_ids = torch.argmax(answers_inf_softmax, dim=-1)
    answers_inf_embeddings = hunch_llm.decoder_embedding(answers_inf_ids.unsqueeze(1)).reshape(batch_size, seq_max_len, -1)
    # attempt_at_answer.shape = (batch_size, seq_len, vocab_size)

    # Compare with the correct answer
    answers_embeddings = hunch_llm.decoder_embedding(answers_ids)
    answer_scores = torch.nn.functional.cosine_similarity(answers_inf_embeddings, answers_embeddings, dim=-1)
    answer_score = answer_scores.mean(-1)

    return answer_score

def rollout(
    # TODO: self.mdl should point to (policy network)
    steps_in_episode,
    nav_agent: ContinuousPolicyGradient,
    hunch_llm: nn.Module,
    env: ITLGraphEnvironment,
    questions_embeddings: torch.Tensor,
    answers_ids: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]: 
    """
    Will execute RL episode rollouts in parallel.
    args:
        kg: Knowledge graph environment.
        num_steps: Number of rollout steps.
        navigator_agent: Policy network.
        graphman: Graph search policy network.
        questions: Questions already pre-embedded to be answered (num_rollouts, question_dim)
        visualize_action_probs: If set, save action probabilities for visualization.
    returns: 
        - log_action_probs: 
        - action_entropy: 
        - path_trace: 
        - path_components:
    """

    assert steps_in_episode > 0

    ########################################
    # Prepare lists to be returned
    ########################################
    log_action_probs = []
    rewards = []

    # Dummy nodes ? TODO: Figur eout what they do.
    # TODO: Perhaps here we can enter through the centroid.
    # For now we still with these dummy
    # NOTE: We repeat these entities until we get the right shape:
    # TODO: make sure we keep all seen nodes up to date

    # Get initial observation. A concatenation of centroid and question atm. Passed through the path encoder
    observations = env.reset(questions_embeddings)
    cur_position, cur_state = observations.position, observations.state
    # Should be of shape (batch_size, 1, hidden_dim)

    # pn.initialize_path(kg) # TOREM: Unecessasry to ask pn to form it for us.
    states_so_far = []
    for t in range(steps_in_episode):

        # Ask the navigator to navigate, agent is presented state, not position
        # State is meant to summrized path history.
        sampled_actions, log_probs, entropies  = nav_agent(cur_state)

        # TODO:Make sure we are gettign rewards from the environment.
        observations = env.step(sampled_actions)

        # For now, we use states given by the path encoder and positions mostly for debugging
        positions, states = (observations.position, observations.state)
        states_so_far.append(states)

        ########################################
        # Calculate the Reward
        ########################################
        stacked_states = torch.stack(states_so_far).permute(1,0,2)
        similarity_scores = calculate_reward(hunch_llm, stacked_states, answers_ids)
        rewards.append(similarity_scores)

        # TODO: Make obseervations not rely on the question

        ########################################
        # Log Stuff for across batch
        ########################################
        log_action_probs.append(log_probs)

        # pn.update_path(action, kg) # TODO: Confirm this is actually needed
        # action_prob = sample_outcome["action_prob"]
        # log_action_probs.append(ops.safe_log(action_prob)) # TODO: Compute this again ( if necessary)
        # action_entropy.append(policy_entropy) # TOREM: Comes from `transit` not sure if I shoudl remove it
        # TODO: Calculate next cur_observation
        
        # TODO: Is this somethign we want?
        # if visualize_action_probs:
        #     top_k_action = sample_outcome["top_actions"]
        #     top_k_action_prob = sample_outcome["top_action_probs"]
        #     path_components.append((e, top_k_action, top_k_action_prob))
        
    return log_action_probs, rewards

def load_qa_data(cached_metadata_path: str, raw_QAData_path, tokenizer_name: str):
    if os.path.exists(cached_metadata_path):
        logger.info(
            f"\033[93m Found cache for the QA data {cached_metadata_path} will load it instead of working on {raw_QAData_path}. \033[0m"
        )
        # Read the first line of the raw csv to count the number of columns
        train_metadata = json.load(open(cached_metadata_path))
        cached_csv_data_path = train_metadata["saved_path"]
        train_df = pd.read_parquet(cached_csv_data_path)
        # Ensure that we are not reading them integers as strings, but also not as floats
        logger.info(f"Loaded cached data from \033[93m\033[4m{json.dumps(cached_metadata_path,indent=4)} \033[0m")
    else:
        logger.info(
            f"\033[93m Did not find cache for the QA data {cached_metadata_path}. Will now process it from {raw_QAData_path} \033[0m"
        )
        text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        train_df, train_metadata = data_utils.process_qa_data( #TOREM: Same here, might want to remove if not really used
            raw_QAData_path,
            cached_metadata_path,
            text_tokenizer,
        )
        logger.info(f"Done. Result dumped at : \n\033[93m\033[4m{train_metadata['saved_path']}\033[0m")
    return train_df, train_metadata

def main():
    # By default we run the config
    # Process data will determine by itself if there is any data to process
    args, tokenizer, logger = initial_setup()

    # TODO: Muybe ? (They use it themselves)
    # initialize_model_directory(args, args.seed)

    ## Agent needs a Knowledge graph as well as the environment
    logger.info(":: Setting up the knowledge graph")
    knowledge_graph = ITLKnowledgeGraph(
        data_dir=args.data_dir,
        model=args.model,
        emb_dropout_rate=args.emb_dropout_rate,
        use_action_space_bucketing=args.use_action_space_bucketing,
        pretrained_embedding_type=args.pretrained_embedding_type,
        pretrained_embedding_weights_path=args.pretrained_embedding_weights_path,
    )

    # Information computed by knowldege graph for future dependency injection
    dim_entity = knowledge_graph.get_entity_dim()
    dim_relation = knowledge_graph.get_relation_dim()
    logger.info("You have reached the exit")

    # Get the Module for Approximate Nearest Neighbor Search
    ########################################
    # Setup the ann index. 
    # Will be needed for obtaining observations.
    ########################################
    logger.info(":: Setting up the ANN Index") 
    ann_index_manager = ANN_IndexMan(
        knowledge_graph.get_all_entity_embeddings_wo_dropout(),
        exact_computation=False,
        nlist=100,
    )

    # Setup the pretrained language model
    logger.info(":: Setting up the pretrained language model")
    config = BartConfig.from_pretrained("facebook/bart-base")
    # Access the hidden size (hidden dimension)
    bart_padding_token_id = config.pad_token_id
    # TODO: Remove the hardcode. Perhaps 
    embedding_hidden_size = config.d_model
    embedding_vocab_size = config.vocab_size
    print(f"The hidden dimension of the embedding layer is {embedding_hidden_size} and its vocab size is {embedding_vocab_size}") 
    hunch_llm = HunchLLM(
        pretrained_transformer_weights_path = args.pretrained_llm_transformer_ckpnt_path,
        xattn_left_dim = args.history_dim,
        llm_model_dim = args.llm_model_dim,
        llm_num_heads = args.llm_num_heads,
        llm_num_layers = args.llm_num_layers,
        llm_ff_dim = args.llm_ff_dim,
        llm_max_seq_length = args.max_seq_length,
        xattn_left_max_seq_length = args.steps_in_episode,
        dropout = args.llm_dropout_rate,
        embedding_padding_id = bart_padding_token_id,
        embedding_dim = embedding_hidden_size,
        embedding_vocab_size = embedding_vocab_size,
    )
    if args.further_train_hunchs_llm:
        # TODO: Ensure we dont have to freeze the model for this.
        hunch_llm.freeze_llm()

    # Setup the entity embedding module
    question_embedding_module = AutoModel.from_pretrained(args.question_embedding_model)
    # Setting up the models
    logger.info(":: Setting up the environment")
    env = ITLGraphEnvironment(
        question_embedding_module=question_embedding_module,
        question_embedding_module_trainable=args.question_embedding_module_trainable,
        entity_dim=dim_entity,
        ff_dropout_rate=args.ff_dropout_rate,
        history_dim=args.history_dim,
        history_num_layers=args.history_num_layers,
        knowledge_graph=knowledge_graph,
        relation_dim=dim_relation,
        ann_index_manager=ann_index_manager,
        steps_in_episode=args.num_rollout_steps
    )

    # Now we load this from the embedding models

    # TODO: Reorganizew the parameters lol
    logger.info(":: Setting up the navigation agent")
    nav_agent = ContinuousPolicyGradient(
        baseline=args.baseline,
        beta=args.beta,
        gamma=args.gamma,
        action_dropout_rate=args.action_dropout_rate,
        action_dropout_anneal_factor=args.action_dropout_anneal_factor,
        action_dropout_anneal_interval=args.action_dropout_anneal_interval,
        num_rollout_steps=args.num_rollout_steps,
        dim_action=dim_relation,
        dim_hidden=args.rnn_hidden,
        dim_observation=args.history_dim, # observation will be into history
    )


    # TODO: Add checkpoint support
    # See args.start_epoch

    ########################################
    # Get the data
    ########################################
    logger.info(":: Setting up the data")
    train_df, train_metadata = load_qa_data(args.cached_QAMetaData_path, args.raw_QAData_path, args.tokenizer_name)

    # TODO: Load the validation data
    # dev_path = os.path.join(args.data_dir, "dev.triples")
    # dev_data = data_utils.load_triples(
    #     dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities
    # )
    # TODO: Make it take check for a checkpoint and decide what start_epoch
    # if args.checkpoint_path is not None:
    #     # TODO: Add it here to load the checkpoint separetely
    #     nav_agent.load_checkpoint(args.checkpoint_path)

    ######## ######## ########
    # Train:
    ######## ######## ########
    start_epoch = 0
    logger.info(":: Training the model")
    train_multihopkg(
        batch_size = args.batch_size,
        epochs = args.epochs,
        nav_agent = nav_agent,
        hunch_llm = hunch_llm,
        learning_rate = args.learning_rate,
        steps_in_episode = args.num_rollout_steps,
        env = env, 
        start_epoch = args.start_epoch,
        train_data = train_df
    )

    # TODO: Evaluation of the model
    # metrics = inference(lf)


if __name__ == "__main__":
    main(),
