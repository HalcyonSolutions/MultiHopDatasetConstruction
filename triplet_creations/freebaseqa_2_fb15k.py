# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:01:52 2024

@author: Eduin Hernandez

Summary: Filters FreebaseQA questions to retain only those compatible with the designated Freebase
    dataset by verifying the presence of the MID entities in both the questions and the target dataset. 
    The script reads the processed Freebase questions, valid triplets, and node data,
    and outputs the filtered questions containing entities that align with the target dataset.
"""
import argparse
from utils.basic import load_triplets, load_pandas

def parse_args() -> argparse.Namespace:
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Filters FreebaseQA questions to only use those present in the given FB15k Compatible Dataset")
    
    # Input
    parser.add_argument('--freebaseqa-path', type=str, default='./data/freebaseqa_unprocessed.csv',
                        help='Path to the CSV file containing processed Jeopardy questions with associated QIDs.')
    parser.add_argument('--triplets-path', type=str, default='./data/triplets_fb15k.txt',
                        help='Path to the text file containing valid triplets of entities used for filtering.')
    parser.add_argument('--train-triplets-path', type=str, default='./data/link_prediction/train_fb15k_237.txt',
                        help='Path to the text file containing training triplets of entities used for filtering.')
    
    # Output
    parser.add_argument('--freebase-output-path', type=str, default='./data/questions/freebaseqa_fb15k.csv',
                        help='Path to save the filtered Jeopardy questions containing entities that match the target dataset.')
    
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    
    #--------------------------------------------------------------------------
    
    fqa_df = load_pandas(args.freebaseqa_path)
    triplet_df = load_triplets(args.triplets_path)
    train_triplet_df = load_triplets(args.train_triplets_path)
    # Create a set of (head, relation, tail) triplets from the training data
    train_triplets = set(
        zip(train_triplet_df['head'], train_triplet_df['relation'], train_triplet_df['tail'])
    )
    
    valid_nodes = set(triplet_df['head']) | set(triplet_df['tail'])
    valid_rels = set(triplet_df['relation'])
    
    # Filter the DataFrame to keep only rows where 'TopicEntityMid' (Query-Entities) and 'AnswersMid' (Answer-Entities) exist in the KG
    valid_df = fqa_df[fqa_df['TopicEntityMid'].isin(valid_nodes) & fqa_df['AnswersMid'].isin(valid_nodes)] # Keep only valid nodes
    valid_df = valid_df[valid_df['InferentialChain'].notna()] # Remove NaN values in InferentialChain

    # Filter the DataFrame to keep only rows where "InferentialChain" (Query-Relations) exit in the KG
    valid_df.loc[:, 'InferentialChain'] = valid_df['InferentialChain'].str.replace('..', ';/').str.replace('.', '/').str.replace(';', '.')
    valid_df.loc[:, 'InferentialChain'] = valid_df['InferentialChain'].apply(lambda x: '/' + x if not x.startswith('/') else x)
    valid_df = valid_df[valid_df['InferentialChain'].isin(valid_rels)]
    
    valid_question_id = set(valid_df['Question-Number'])
    
    print(f"Number of Unique Questions: {len(valid_question_id)}")
    print(f"Number of Questions: {len(valid_df['Question-Number'])}")

    # Add paths, label of train if present in the training set, and hops
    valid_df['Paths'] = valid_df.apply(lambda x: f"[['{x['TopicEntityMid']}', '{x['InferentialChain']}', '{x['AnswersMid']}']]", axis=1)
    valid_df['Hops'] = 1

    # First pass: Tentative label for each row
    valid_df['SplitLabel'] = valid_df.apply(
        lambda row: "train" if (row['TopicEntityMid'], row['InferentialChain'], row['AnswersMid']) in train_triplets else "test",
        axis=1
    )

    # Second pass: Get all (Question-Number, TopicEntityMid) pairs that are marked as 'train'
    train_qid_topic_pairs = set(
        valid_df.loc[valid_df['SplitLabel'] == 'train', ['Question-Number', 'TopicEntityMid']].itertuples(index=False, name=None)
    )

    # Apply the rule: if any path in the same (Question-Number, TopicEntityMid) group is train, all must be train
    valid_df['SplitLabel'] = valid_df.apply(
        lambda row: "train" if (row['Question-Number'], row['TopicEntityMid']) in train_qid_topic_pairs else "test",
        axis=1
    )

    # Rename columns to match the desired output format
    valid_df.rename(columns={'TopicEntityMid': 'Query-Entity'}, inplace=True)
    valid_df.rename(columns={'TopicEntityName': 'Query-Entity-Title'}, inplace=True)
    valid_df.rename(columns={'InferentialChain': 'Query-Relation'}, inplace=True)
    valid_df.rename(columns={'AnswersMid': 'Answer-Entity'}, inplace=True)
    valid_df.rename(columns={'AnswersName': 'Answer-Entity-Title'}, inplace=True)
    valid_df.rename(columns={'Question-Number': 'Question-ID'}, inplace=True)

    valid_df = valid_df.reset_index(drop=True)
    valid_df['Question-Number'] = valid_df.index + 1

    print(valid_df['SplitLabel'].value_counts(normalize=True) * 100)

    # Save the filtered DataFrame to a CSV file
    valid_df.to_csv(args.freebase_output_path, index=False)

    # Retain only the relevant columns
    clean_df = valid_df[['Question-Number', 'Question', 'Answer', 'Hops', 'Query-Entity', 'Query-Relation', 'Answer-Entity',
                          'Query-Entity-Title', 'Answer-Entity-Title', 'Paths', 'SplitLabel']]
    clean_df.to_csv(args.freebase_output_path.replace('.csv','') + '_clean.csv', index=False)
