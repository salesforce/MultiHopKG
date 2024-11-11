"""
Instead of using bach config filees, we will just separate them into different files we can just import

This is config `alpha.py`:
Just to try to get the first run running.
"""

import argparse
import os
import sys


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    # For data processing
    path_to_running_file = os.path.abspath(sys.argv[0])
    default_cache_dir = os.path.join(os.path.dirname(path_to_running_file), ".cache")
    ap.add_argument(
        "--QAtriplets_raw_dir",
        type=str,
        default=os.path.join(path_to_running_file, "data/itl/multihop_ds_datasets_FbWiki_TriviaQA.csv"),
    )
    ap.add_argument(
        "--QAtriplets_cache_dir",
        type=str,
        default=os.path.join(default_cache_dir, "qa_triplets.csv"),
    )
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=420, metavar="S")
    ap.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")

    # New Paremters introduced by the new model
    ap.add_argument("--pretrained_embedding_type",type=str,default="conve",help="The type of pretrained embedding to use")
    ap.add_argument("--pretrained_embedding_weights_path",type=str,default="./models/itl/pretrained_embeddings.tar",help="Theh path to the pretrained embedding weights")
    ap.add_argument("--emb_dropout_rate", type=float, default=0.3, help='Knowledge graph embedding dropout rate (default: 0.3)')
    # ap.add_argument("--relation_only",  action="store_true",  help='search with relation information only, ignoring entity representation (default: False)')
    # ap.add_argument('--history_dim', type=int, default=400, metavar='H',
    ap.add_argument('--history_dim', type=int, default=768, metavar='H',
                        help='action history encoding LSTM hidden states dimension (default: 400)')
    ap.add_argument('--history_num_layers', type=int, default=3, metavar='L',
                        help='action history encoding LSTM number of layers (default: 1)')
    ap.add_argument('--ff_dropout_rate', type=float, default=0.1,
                        help='Feed-forward layer dropout rate (default: 0.1)')
    ap.add_argument('--xavier_initialization', type=bool, default=True,
                        help='Initialize all model parameters using xavier initialization (default: True)')
    ap.add_argument('--epochs',type=int,default=200,help='Epochs for training')
    # TODO: tinker with this value
    ap.add_argument('--rnn_hidden',type=int,default=400,help='RNN hidden dimension')
    ap.add_argument('--raw_QAData_path', type=str, default="data/itl/multihop_ds_datasets_FbWiki_TriviaQA.parquet", help="Directory where the QA knowledge graph data is stored (default: None)")
    ap.add_argument('--cached_QAMetaData_path', type=str, default="./.cache/itl/itl_data-tok_bert-base-uncased-maxpathlen_5.json", help="Path for precomputed QA knowledge graph data. Precomputing is mostly tokenizaiton.")
    # TODO: (eventually) We might want to add option of locally trained models.
    ap.add_argument('--question_embedding_model', type=str, default="bert-base-uncased", help="The Question embedding model to use (default: bert-base-uncased)")
    ap.add_argument('--question_embedding_module_trainable', type=bool, default=True, help="Whether the question embedding model is trainable or not (default: True)")
    ap.add_argument('--exact_nn',  action="store_true", help="Whether to use exact nearest neighbor search or not (default: False)")
    ap.add_argument('--num_cluster_for_ivf', type=int, default=100, help="Number of clusters for the IVF index if exact_computation is False (default: 100)")
    ap.add_argument('--further_train_hunchs_llm',  action="store_true", help="Whether to further pretrain the language model or not (default: False)")
    ap.add_argument('--pretrained_llm_for_hunch', type=str, default="bert-base-uncased", help="The pretrained language model to use (default: bert-base-uncased)")
    ap.add_argument('--pretrained_llm_transformer_ckpnt_path', type=str, default="models/itl/pretrained_transformer_e1_s9176.ckpt", help="The path to the pretrained language model transformer weights (default: models/itl/pretrained_transformer_e1_s9176.ckpt)")

    # Wand DB Modell
    ap.add_argument("-w", "--wandb", action="store_true")
    ap.add_argument("--wandb_project_name", help="wandb: Project name label", type=str)
    ap.add_argument("--wr_name", help="wandb: Run Namel label", type=str)
    ap.add_argument("--wr_notes", help="wand:  Run Notes", type=str)

    # These are based on Halcyon/FoundationalLanguageModel
    # TODO: We should have checkpoitns have this informaiton encoded in them.
    ap.add_argument("--llm_model_dim", default=768)
    ap.add_argument("--llm_num_heads", default=8)
    ap.add_argument("--llm_num_layers", default=3)
    ap.add_argument("--llm_ff_dim", default=3072)
    ap.add_argument("--llm_ff_dropout_rate", default=0.1)
    ap.add_argument("--llm_dropout_rate", default=0.1)
    ap.add_argument("--max_seq_length", default=1024)
    ap.add_argument('--batches_b4_eval', type=int, default=1, help='Number of batches to run before evaluation (default: 100)')


    # NOTE: Legacy Parameters
    # Might want to get rid of them as we see fit.
    ap.add_argument('--relation_only_in_path', action='store_true',
                        help='include intermediate entities in path (default: False)')
        
    ap.add_argument('--run_analysis', action='store_true',
                    help='run algorithm analysis and print intermediate results (default: False)')
    ap.add_argument('--data_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'),
                    help='directory where the knowledge graph data is stored (default: None)')
    ap.add_argument('--model_root_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model'),
                    help='root directory where the model parameters are stored (default: None)')
    ap.add_argument('--model_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model'),
                    help='directory where the model parameters are stored (default: None)')
    ap.add_argument('--model', type=str, default='point',
                    help='knowledge graph QA model (default: point)')
    ap.add_argument('--use_action_space_bucketing', action='store_true',
                    help='bucket adjacency list by outgoing degree to avoid memory blow-up (default: False)')
    ap.add_argument('--train_entire_graph', type=bool, default=False,
                    help='add all edges in the graph to extend training set (default: False)')
    ap.add_argument('--num_epochs', type=int, default=200,
                    help='maximum number of pass over the entire training set (default: 20)')
    ap.add_argument('--num_wait_epochs', type=int, default=5,
                    help='number of epochs to wait before stopping training if dev set performance drops')
    ap.add_argument('--num_peek_epochs', type=int, default=2,
                    help='number of epochs to wait for next dev set result check (default: 2)')
    ap.add_argument('--start_epoch', type=int, default=0,
                    help='epoch from which the training should start (default: 0)')
    ap.add_argument('--batch_size', type=int, default=256,
                    help='mini-batch size (default: 256)')
    ap.add_argument('--train_batch_size', type=int, default=256,
                    help='mini-batch size during training (default: 256)')
    ap.add_argument('--dev_batch_size', type=int, default=64,
                    help='mini-batch size during inferece (default: 64)')
    ap.add_argument('--learning_rate', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
    ap.add_argument('--learning_rate_decay', type=float, default=1.0,
                    help='learning rate decay factor for the Adam optimizer (default: 1)')
    ap.add_argument('--adam_beta1', type=float, default=0.9,
                    help='Adam: decay rates for the first movement estimate (default: 0.9)')
    ap.add_argument('--adam_beta2', type=float, default=0.999,
                    help='Adam: decay rates for the second raw movement estimate (default: 0.999)')
    ap.add_argument('--grad_norm', type=float, default=10000,
                    help='norm threshold for gradient clipping (default 10000)')
    ap.add_argument('--action_dropout_rate', type=float, default=0.1,
                    help='Dropout rate for randomly masking out knowledge graph edges (default: 0.1)')
    ap.add_argument('--action_dropout_anneal_factor', type=float, default=0.95,
	                help='Decrease the action dropout rate once the dev set results stopped increase (default: 0.95)')
    ap.add_argument('--action_dropout_anneal_interval', type=int, default=1000,
		            help='Number of epochs to wait before decreasing the action dropout rate (default: 1000. Action '
                         'dropout annealing is not used when the value is >= 1000.)')
    ap.add_argument('--steps_in_episode', type=int, default=20,
                    help='number of steps in episode (default: 20)')
    ap.add_argument('--num_rollout_steps', type=int, default=3,
                    help='maximum path length (default: 3)')
    ap.add_argument('--beta', type=float, default=0.0,
                    help='entropy regularization weight (default: 0.0)')
    ap.add_argument('--gamma', type=float, default=1,
                    help='moving average weight (default: 1)')
    ap.add_argument('--baseline', type=str, default='n/a',
                    help='baseline used by the policy gradient algorithm (default: n/a)')
    ap.add_argument('--beam_size', type=int, default=100,
                    help='size of beam used in beam search inference (default: 100)')


    return ap.parse_args()

