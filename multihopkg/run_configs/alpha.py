"""
Instead of using bach config filees, we will just separate them into different files we can just import
"""

from argparse import ArgumentParser
import os
import sys


def get_args():
    ap = ArgumentParser()

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
    ap.add_argument("--tokenizer", type=str, default="bert-base-uncased")

    # NOTE: Legacy Parameters
    # Might want to get rid of them as we see fit.
    ap.add_argument("--relation_only", type=str, default="", help="")
    ap.add_argument("--history_dim", type=str, default="", help="")
    ap.add_argument("--history_num_layers", type=str, default="", help="")
    ap.add_argument("--entity_dim", type=str, default="", help="")
    ap.add_argument("--relation_dim", type=str, default="", help="")
    ap.add_argument("--ff_dropout_rate", type=str, default="", help="")
    ap.add_argument("--xavier_initialization", type=str, default="", help="")
    ap.add_argument("--relation_only_in_path", type=str, default="", help="")

    return ap.parse_args()
