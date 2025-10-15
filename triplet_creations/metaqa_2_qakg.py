import argparse
import os

import pandas as pd

from utils.basic import load_triplets, load_pandas, save_triplets

# TODO: Cleanup & Move functions to utils

def parse_args() -> argparse.Namespace:
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Processes the MetaQA dataset to a compatible format for MultiHop agents.")
    
    # Input
    parser.add_argument('--metaqa-path', type=str, default='./data/source/MetaQA/',
                        help='Folder path containing the MetaQA.')
    
    # Output
    parser.add_argument('--triplet-path', type=str, default='./data/link_prediction/MetaQA/',
                        help='')
    parser.add_argument('--question-path', type=str, default='./data/qa/MetaQA/',
                        help='')

    return parser.parse_args()

def extract_question_entity(df):
    return df['Question'].str.extract(r'\[(.*?)\]')

def extract_answer_entity(df):
    return df['Answer'].str.split('|')

def remove_brackets(text: str) -> str:
    return text.replace('[', '').replace(']', '')

def merge_vanilla_ntm_questions(vanilla_path: str, ntm_path: str, split: str = 'train'):
    qa_vanilla = load_pandas(os.path.join(vanilla_path, f'qa_{split}.txt'), sep='\t', header=None, names=['Question', 'Answer'])
    qa_ntm = load_pandas(os.path.join(ntm_path, f'qa_{split}.txt'), sep='\t', header=None, names=['Question_Alt', 'Answer'])
    qa = qa_vanilla.merge(qa_ntm['Question_Alt'], left_index=True, right_index=True)
    qa['SplitLabel'] = split
    assert len(qa) == len(qa_ntm) == len(qa_vanilla), f"Merging alternative questions failed! New length: {len(qa)}, Vanilla length: {len(qa_vanilla)}, NTM length: {len(qa_ntm)}"
    return qa

def process_qa(df):
    # Extract Entities
    df['Source-Entity'] = extract_question_entity(df).map(str.lower)
    df['Answer'] = extract_answer_entity(df)

    df['Question'] = df['Question'].apply(remove_brackets)
    df['Question_Alt'] = df['Question_Alt'].apply(remove_brackets)

    df['Answer-Entity'] = df['Answer'].apply(lambda x: [i.lower() for i in x] if isinstance(x, list) else x)
    
    # # Separate multiple answers into different rows
    # df = df.explode('Answer').reset_index(drop=True)
    # df['Answer-Entity'] = df['Answer'].map(str.lower)
    # lower each string in the list
    # df['Answer_Alt'] = df['Answer_Alt'].apply(lambda x: [i.lower() for i in x] if isinstance(x, list) else x)
    # df['Answer-Entity_Alt'] = df['Answer_Alt'].apply(lambda x: x.lower()) #.map(str.lower)

    return df

if __name__ == '__main__':

    args = parse_args()

    kg_path = os.path.join(args.metaqa_path, 'kb.txt')

    kg_graph = load_triplets(kg_path, sep='|')
    kg_graph = kg_graph.map(str.lower)
    save_triplets(kg_graph, args.triplet_path + 'triplets.txt')
    relations = set(kg_graph['relation'])
    entities = set(kg_graph['head']).union(set(kg_graph['tail']))
    print(f"Relations: {relations}")

    nhop_qa = []
    for i0 in range(1, 4):
        vanilla_path = os.path.join(args.metaqa_path, f'{i0}-hop/vanilla/')
        ntm_path = os.path.join(args.metaqa_path, f'{i0}-hop/ntm/')

        train_qa = merge_vanilla_ntm_questions(vanilla_path, ntm_path, split='train')
        dev_qa = merge_vanilla_ntm_questions(vanilla_path, ntm_path, split='dev')
        test_qa = merge_vanilla_ntm_questions(vanilla_path, ntm_path, split='test')

        full_qa = pd.concat([train_qa, dev_qa, test_qa], ignore_index=True)
        full_qa['Hops'] = i0

        full_qa = process_qa(full_qa)
        full_qa = full_qa[['Question', 'Question_Alt', 'Answer', 'Hops', 'Source-Entity', 'Answer-Entity', 'SplitLabel']]
        print(full_qa.head(3))

        nhop_qa.append(full_qa)
        # full_qa.to_csv(os.path.join(args.metaqa_path, f'metaqa_{i0}hop.csv'), index=False)

    nhop_qa = pd.concat(nhop_qa, ignore_index=True)

    # Add Question Number
    nhop_qa.insert(0, 'Question-Number', range(0, len(nhop_qa)))
    nhop_qa.to_csv(os.path.join(args.question_path, 'metaqa_nhop.csv'), index=False)

    for i0 in range(1, 4):
        nhop_qa_subset = nhop_qa[nhop_qa['Hops'] == i0]
        nhop_qa_subset.to_csv(os.path.join(args.question_path, f'metaqa_{i0}hop.csv'), index=False)

    # check that Source-Entity and Answer-Entity are in the entities set
    assert nhop_qa['Source-Entity'].isin(entities).all(), "Some Source-Entities are not in the KG entities!"
    assert nhop_qa['Answer-Entity'].apply(lambda ans_list: all(ans in entities for ans in ans_list)).all(), "Some Answer-Entities are not in the KG entities!"
    # assert nhop_qa['Answer-Entity'].isin(entities).all(), "Some Answer-Entities are not in the KG entities!"
    print("All Source-Entities and Answer-Entities are present in the KG entities.")
