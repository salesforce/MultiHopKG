"""
LG: A place holder for the data preparation script

Explanation:
    I dont like all this being muddled in the main script.
    So, we ended up with a router for different data operations. 
    But, still we can run all preprocessing in one go with the `all` function.
"""
import argparse
import os
from multihopkg import data_utils 
from transformers import AutoTokenizer
from multihopkg.logging import setup_logger
from multihopkg.utils.setup import get_git_root

def process_traditional_kb_data(data_dir:str, test:bool, model:str, add_reverse_relations: bool):
    # NOTE: Their code here
    raw_kb_path = os.path.join(data_dir, 'raw.kb')
    train_path = data_utils.get_train_path(data_dir, test,model)
    dev_path = os.path.join(data_dir, 'dev.triples')
    test_path = os.path.join(data_dir, 'test.triples')
    data_utils.prepare_kb_envrioment(raw_kb_path, train_path, dev_path, test_path, test, add_reverse_relations)


def process_qa_data(raw_data_dir: str,  cache_data_dir: str, text_tokenizer: str):
    # Load the Transformers Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(text_tokenizer)
    # Load the data
    logger.info(
        "Loading the data with parameters:\n"
        f"---> raw_data_dir: {raw_data_dir}\n"
        f"---> cache_data_dir: {cache_data_dir}\n"
        f"---> text_tokenizer: {text_tokenizer}\n"
    )
    data_utils.process_qa_data(
        raw_data_dir,
        cache_data_dir,
        tokenizer,
    )


def all_arguments(valid_operations: list) -> argparse.Namespace:

    ap = argparse.ArgumentParser()
    #-----------------
    # Common Arguments
    #-----------------
    ap.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="logging level (default: INFO)",
    )

    # Rather than __file__ in case we move the script (and clarity)
    repo_root = get_git_root()
    assert repo_root is not None, "Could not find the root of the git repository"

    #----------------------------
    # process_traditional_kb_data
    #----------------------------
    ap.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(repo_root, "data"),
        help="directory where the knowledge graph data is stored (default: None)",
    )
    ap.add_argument(
        "--test",
        action="store_true",
        help="perform inference on the test set (default: False)",
    )
    ap.add_argument(
        "--add_reverse_relations",
        action="store_true",
        help="add reverse relations to KB (default: False)",
    )

    #----------------
    # process_qa_data
    #----------------
    # NOTE: we use __file__ as a proxy for repo root
    ap.add_argument(
        "--raw_QAPathData_path",
        type=str,
        default=os.path.join(repo_root, "data/itl/multihop_ds_datasets_FbWiki_TriviaQA.csv"),
        help="Directory where the knowledge graph data is stored (default: None)",
    )
    ap.add_argument(
        "--cached_QAPathData_path",
        type=str,
        default=os.path.join(repo_root, ".cache/itl/itl_data-tok_{}-maxpathlen_{}.csv"),
        help="Directory where the knowledge graph data is stored (default: None)",
    )
    ap.add_argument(
        "--text_tokenizer",
        type=str,
        default="bert-base-uncased",
        help="Directory where the knowledge graph data is stored (default: None)",
    )


    #----------------------------------
    # For choosing operaiton to perform
    #----------------------------------
    ap.add_argument(
        "--operation",
        type=str,
        required=True,
        choices=valid_operations,
        help="operation to perform",
    )

    args = ap.parse_args()

    #---------------------
    # Some Sanity Checking
    #---------------------
    os.makedirs(os.path.dirname(args.cached_QAPathData_path), exist_ok=True)

    return args

def main(args: argparse.Namespace, valid_operations: dict):

    args = all_arguments(list(valid_operations.keys()))

    if args.operation not in valid_operations:
        raise ValueError("Invalid operation: {}".format(args.operation))

    # Switch on the operation
    operation = valid_operations[args.operation]
    if args.operation == "process_data":
        operation(args.data_dir, args.test, args.model, args.add_reverse_relations)
    elif args.operation == "process_qa_data":
        operation(args.raw_QAPathData_path, args.cached_QAPathData_path, args.text_tokenizer)
    else:
        raise NotImplementedError

def all(args: argparse.Namespace):
    # First prepare the traditinal KB data
    process_traditional_kb_data(args.data_dir, args.test, args.model, args.add_reverse_relations)
    # Then prepare the QA data
    process_qa_data(args.raw_QAPathData_path, args.cached_QAPathData_path, args.text_tokenizer)

if __name__ == "__main__":
    adadaddadad = {
        "process_traditional_kb_data": process_traditional_kb_data,
        "process_qa_data": process_qa_data,
    }
    # Want to aboid these to be global
    xdaodijadoj = all_arguments(list(adadaddadad.keys()))
    logger = setup_logger(
        logger_name=os.path.basename(__file__).replace(".py", ""),
        logging_level=xdaodijadoj.logging_level,
    )


    main(args=xdaodijadoj, valid_operations=adadaddadad)
