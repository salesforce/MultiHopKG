#!/usr/bin/env bash

data_dir="data/umls"
model="point.rs.conve"
group_examples_by_query="False"
use_action_space_bucketing="True"

bandwidth=400
entity_dim=200
relation_dim=200
history_dim=200
history_num_layers=3
num_rollouts=20
num_rollout_steps=2
bucket_interval=10
num_epochs=1000
num_wait_epochs=200
num_peek_epochs=2
batch_size=128
train_batch_size=128
dev_batch_size=64
learning_rate=0.001
baseline="n/a"
grad_norm=5
emb_dropout_rate=0.1
ff_dropout_rate=0
action_dropout_rate=0.95
action_dropout_anneal_interval=1000
reward_shaping_threshold=0
beta=0.05
relation_only="False"
beam_size=128

# pre-trained embedding based model checkpoints
distmult_state_dict_path="model/umls-distmult-xavier-200-200-0.003-0.3-0.1/model_best.tar"
complex_state_dict_path="model/umls-complex-xavier-200-200-0.003-0.3-0.1/model_best.tar"
conve_state_dict_path="model/umls-conve-RV-xavier-200-200-0.003-32-3-0.3-0.3-0.2-0.1/model_best.tar"

num_paths_per_entity=-1
margin=-1
