"""
LG: A place holder for the data preparation script
I dont like all this being muddled in the main script.
"""
import argparse
import os
from multihopkg import data_utils 

def process_data(args: argparse.Namespace):
    # NOTE: Their code here
    data_dir = args.data_dir
    raw_kb_path = os.path.join(data_dir, 'raw.kb')
    train_path = data_utils.get_train_path(args)
    dev_path = os.path.join(data_dir, 'dev.triples')
    test_path = os.path.join(data_dir, 'test.triples')
    data_utils.prepare_kb_envrioment(raw_kb_path, train_path, dev_path, test_path, args.test, args.add_reverse_relations)

def main():
    valid_opereations = ["process_data"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'),
                    help='directory where the knowledge graph data is stored (default: None)')
    ap.add_argument("--test", action='store_true',
                    help='perform inference on the test set (default: False)')
    ap.add_argument("--add_reverse_relations", action='store_true',
                    help='add reverse relations to KB (default: False)')
    # Do a multiple choice option that is required
    ap.add_argument("--operation", type=str, default="process_data", choices=valid_opereations,
                    help='operation to perform')

    args = ap.parse_args()
    if args.operation not in valid_opereations:
        raise ValueError("Invalid operation: {}".format(args.operation))

    # Switch on the operation
    if args.operation == "process_data":
        process_data(args)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
