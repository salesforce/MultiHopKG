#!/usr/bin/env bash

data_dir="data/umls"
model="distmult"
add_reversed_training_edges="False"
group_examples_by_query="True"
entity_dim=200
relation_dim=200
num_rollouts=5
bucket_interval=10
num_epochs=1
num_wait_epochs=400
batch_size=512
train_batch_size=512
dev_batch_size=64
learning_rate=0.003
grad_norm=5
emb_dropout_rate=0.3
beam_size=128
num_negative_samples=20
margin=10
