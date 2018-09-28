#!/usr/bin/env bash

data_dir="data/NELL-995"
model="conve"
add_reversed_training_edges="True"
group_examples_by_query="True"
entity_dim=200
relation_dim=200
num_rollouts=1
bucket_interval=10
num_epochs=1000
num_wait_epochs=500
batch_size=512
train_batch_size=512
dev_batch_size=128
learning_rate=0.003
grad_norm=5
emb_dropout_rate=0.3
beam_size=128

num_negative_samples=100
margin=0.5
