#!/usr/bin/env bash

data_dir="data/NELL-995"
model="point"
group_examples_by_query="False"
use_action_space_bucketing="True"

bandwidth=256
entity_dim=200
relation_dim=200
history_dim=200
history_num_layers=3
num_rollouts=20
num_rollout_steps=3
num_epochs=1000
num_wait_epochs=100
num_peek_epochs=2
bucket_interval=5
batch_size=128
train_batch_size=128
dev_batch_size=16
learning_rate=0.001
baseline="n/a"
grad_norm=5
emb_dropout_rate=0.3
ff_dropout_rate=0.1
action_dropout_rate=0.3
action_dropout_anneal_interval=1000
beta=0.05
relation_only="False"
beam_size=128

num_paths_per_entity=-1
margin=-1
