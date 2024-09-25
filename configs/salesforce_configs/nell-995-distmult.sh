#!/usr/bin/env bash

data_dir="data/NELL-995"
model="distmult"
group_examples_by_query="True"
entity_dim=200
relation_dim=200
num_rollouts=1
bucket_interval=10
num_epochs=1000
num_wait_epochs=300
batch_size=512
train_batch_size=512
dev_batch_size=64
learning_rate=0.003
grad_norm=5
emb_dropout_rate=0.3
beam_size=100
num_negative_samples=50
margin=10
