"""
Will take a (c/s/etc)sv file and create a file with the entity/relationship information
It can have any amount of colums, it will just take the first with a Q/R prefix as the one to use
"""

import argparse
from enum import Enum
from re import I

import pandas as pd

from utils.wikidata_v2 import process_data_batch_generic, process_relationship_data, process_entity_triplets


class IdType(Enum):
    ENTITY = 1
    RELATIONSHIP = 2


def arguments() -> argparse.Namespace:

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv_file_path", type=str, required=True)
    ap.add_argument("--output_csv_file_path", type=str, required=True)
    ap.add_argument("--sep", "-s", default="\t", type=str)
    ap.add_argument("--max_workers", "-w", default=4, type=int)
    ap.add_argument("--num_rows", "-n", default=None, type=int)
    ap.add_argument(
        "--id_type",
        "-t",
        required=True,
        choices=["entity", "relationship"],
        type=str,
    )
    ap.add_argument(
        "--batch_size",
        "-b",
        default=50,
        type=int,
        help="What batch size to use when querying wikidata endpoing with SPARQL",
    )
    ap.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed requests. Defaults to 3.",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=2,
        help="Timeout in seconds for each request. Defaults to 2 seconds.",
    )
    ap.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="Flag to enable verbose output (e.g., print errors during processing).",
    )
    ap.add_argument("--failed_log_path", type=str, default="./failed_log.log")

    return ap.parse_args()


def _determine_id_column(idtype: IdType, data: pd.DataFrame) -> int:
    first_row: pd.Series = data.iloc[0, :]
    print(f"first row looks like {first_row.values}")
    for i, elem in enumerate(first_row):
        if not isinstance(elem, str):
            continue
        if idtype == IdType.ENTITY and elem[0] == "Q":
            return i
        elif idtype == IdType.RELATIONSHIP and elem[0] == "P":
            return i
    raise ValueError("IdType not recognized")


def main():
    args = arguments()

    # Read initial data
    data = pd.read_csv(args.input_csv_file_path, sep=args.sep, header=None)
    print(f"Loaded data with {len(data)} elements and columns {data.columns}")
    id_type = IdType.ENTITY if args.id_type == "entity" else IdType.RELATIONSHIP
    column_id = _determine_id_column(id_type, data)
    print(f"Determined that the column with the ids is {column_id} (0-based indexing)")

    ids_set = data[column_id].tolist()

    if args.num_rows:
        ids_set = ids_set[:args.num_rows]   

    if id_type == IdType.ENTITY:
        df = process_data_batch_generic(
            id_list=ids_set,
            is_entity=True,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            timeout=args.timeout,
            verbose=args.verbose,
            failed_log_path=args.failed_log_path,
        )
    elif id_type == IdType.RELATIONSHIP:
        df = process_data_batch_generic(
            id_list=ids_set,
            is_entity=False,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            timeout=args.timeout,
            verbose=args.verbose,
            failed_log_path=args.failed_log_path,
        )
    else:
        raise ValueError(f"Invalid id_type {args.id_type}")


    assert df is not None
    df.to_csv(args.output_csv_file_path, index=False)
    print(f"Data was succesfully saved to: {args.output_csv_file_path}")


if __name__ == "__main__":
    main()
