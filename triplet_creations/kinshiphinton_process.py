"""

"""
import argparse
import ast
import pandas as pd
from utils.basic import load_triplets, load_pandas

def parse_args() -> argparse.Namespace:
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Processes the Kinship Hinton dataset to filter questions based on training triplets.")
    
    # Input
    parser.add_argument('--kinship-path', type=str, default='./data/kinship_hinton_qa_1hop.csv',
                        help='')
    parser.add_argument('--train-triplets-path', type=str, default='./data/link_prediction/train_kinshiphinton.txt',
                        help='')
    
    # Output
    parser.add_argument('--kinship-output-path', type=str, default='./data/questions/kinship_hinton_qa_1hop.csv',
                        help='')
    
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    
    #--------------------------------------------------------------------------
    
    qa_df = load_pandas(args.kinship_path)
    train_triplet_df = load_triplets(args.train_triplets_path)

        # Create a set of (head, relation, tail) triplets from the training data
    train_triplets = set(
        zip(train_triplet_df['head'], train_triplet_df['relation'], train_triplet_df['tail'])
    )

    # Convert string representations of lists to actual lists
    qa_df['Relevant-Entities'] = qa_df['Relevant-Entities'].apply(ast.literal_eval)
    qa_df['Relevant-Relations'] = qa_df['Relevant-Relations'].apply(ast.literal_eval)
    
    # Extract the single string from each list
    qa_df['Relevant-Entities'] = qa_df['Relevant-Entities'].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)
    qa_df['Relevant-Relations'] = qa_df['Relevant-Relations'].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

    # First pass: Tentative label for each row
    qa_df['SplitLabel'] = qa_df.apply(
        lambda row: "train" if (row['Relevant-Entities'], row['Relevant-Relations'], row['Answer-Entity']) in train_triplets else "test",
        axis=1
    )

    print(qa_df['SplitLabel'].value_counts(normalize=True) * 100)

    # Renaming
    qa_df.rename(columns={'Path': 'Paths'}, inplace=True)
    qa_df.rename(columns={'Relevant-Entities': 'Source-Entity'}, inplace=True)
    qa_df.rename(columns={'Relevant-Relations': 'Query-Relation'}, inplace=True)

    qa_df = qa_df[['Question-Number', 'Question', 'Answer', 'Hops', 'Source-Entity', 'Query-Relation', 'Answer-Entity',
                   'Paths', 'SplitLabel']]
    
    qa_df.to_csv(args.kinship_output_path, index=False)