"""
Simple utility for splitting your datasets. 
Run it: `python -m multihopkg.utils.data_splitting <args...>`
"""

import pandas as pd
import argparse
import os
import sys
from sklearn.model_selection import train_test_split


def split_dataset(file_path, train_size, test_size, val_size):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Path without basename
    file_path = "/".join(file_path.split("/")[:-1])

    # Ensure the split sizes sum to 1
    if round(args.train_size + args.test_size + args.val_size, 2) != 1.0:
        print("Error: The sum of train, test, and validation sizes must be 1.0")
        sys.exit(1)

    # Split the data
    train_data, temp_data = train_test_split(df, test_size=(1 - train_size))
    val_data, test_data = train_test_split(
        temp_data, test_size=(test_size / (test_size + val_size))
    )

    # Save the splits to CSV files
    train_data.to_csv(os.path.join(file_path, "train.csv"), index=False)  # type: ignore
    val_data.to_csv(os.path.join(file_path, "dev.csv"), index=False)  # type: ignore
    test_data.to_csv(os.path.join(file_path, "test.csv"), index=False)  # type:ignore

    print(
        "Data has been split and saved to train_data.csv, val_data.csv, and test_data.csv"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a dataset into train, test, and validation sets."
    )
    parser.add_argument("file_path", type=str, help="Path to the CSV file")
    parser.add_argument(
        "--train_size",
        type=float,
        help="Percentage of data to be used for training (e.g., 0.7 for 70%)",
        default=0.7,
    )
    parser.add_argument(
        "--test_size",
        type=float,
        help="Percentage of data to be used for testing (e.g., 0.2 for 20%)",
        default=0.2,
    )
    parser.add_argument(
        "--val_size",
        type=float,
        help="Percentage of data to be used for validation (e.g., 0.1 for 10%)",
        default=0.1,
    )

    args = parser.parse_args()

    split_dataset(args.file_path, args.train_size, args.test_size, args.val_size)
