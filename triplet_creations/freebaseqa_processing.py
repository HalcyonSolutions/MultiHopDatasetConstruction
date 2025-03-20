# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 19:58:50 2024

@author: Eduin Hernandez
"""
import json
import pandas as pd
import glob
# import os

from utils.basic import load_triplets

def read_json_to_dataframe(file_paths):
    # Initialize lists to store data for DataFrame
    question_ids = []
    raw_questions = []
    processed_questions = []
    parse_ids = []
    potential_topic_entity_mentions = []
    topic_entity_names = []
    topic_entity_mids = []
    inferential_chains = []
    answers_mids = []
    answers_names = []
    file_sources = []

    # Iterate over all files and extract data
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Determine the source type from the file name
        if "train" in file_path:
            source = "train"
        elif "dev" in file_path:
            source = "dev"
        elif "eval" in file_path:
            source = "eval"
        elif "partial" in file_path:
            source = "partial"
        else:
            source = "unknown"

        # Extract questions and related information from JSON
        for question in data['Questions']:
            question_id = question['Question-ID']
            raw_question = question['RawQuestion']
            processed_question = question['ProcessedQuestion']

            for parse in question['Parses']:
                parse_id = parse['Parse-Id']
                potential_topic_entity_mention = parse.get('PotentialTopicEntityMention', None)
                topic_entity_name = parse.get('TopicEntityName', None)
                topic_entity_mid = parse.get('TopicEntityMid', None)
                inferential_chain = parse.get('InferentialChain', None)

                for answer in parse['Answers']:
                    answer_mid = answer.get('AnswersMid', None)
                    answer_name = answer.get('AnswersName', None)

                    # Append extracted data to corresponding lists
                    question_ids.append(question_id)
                    raw_questions.append(raw_question)
                    processed_questions.append(processed_question)
                    parse_ids.append(parse_id)
                    potential_topic_entity_mentions.append(potential_topic_entity_mention)
                    topic_entity_names.append(topic_entity_name)
                    topic_entity_mids.append('/' + topic_entity_mid.replace('.', '/'))
                    inferential_chains.append(inferential_chain)
                    answers_mids.append('/' + answer_mid.replace('.', '/'))
                    answers_names.append(answer_name[0])
                    file_sources.append(source)

    # Create DataFrame from the extracted data
    dataframe = pd.DataFrame({
        'Question-Number': question_ids,
        'Question': raw_questions,
        'Answer': answers_names,
        'RawQuestion': raw_questions,
        'ProcessedQuestion': processed_questions,
        'Parse-Id': parse_ids,
        'PotentialTopicEntityMention': potential_topic_entity_mentions,
        'TopicEntityName': topic_entity_names,
        'TopicEntityMid': topic_entity_mids,
        'InferentialChain': inferential_chains,
        'AnswersMid': answers_mids,
        'AnswersName': answers_names,
        'Source': file_sources
    })

    return dataframe

# Example usage
if __name__ == "__main__":
    file_paths = glob.glob("./data/FreebaseQA-*.json")  # Change the pattern as needed to include desired files
    
    df = read_json_to_dataframe(file_paths)
    entities = set(df['TopicEntityMid']) | set(df['AnswersMid'])
    rels = set(df['InferentialChain'])
    question_id = set(df['Question-Number'])
    
    node_df = df[['TopicEntityMid', 'TopicEntityName']].rename(columns={'TopicEntityMid': 'MID',
                            'TopicEntityName': 'Title'})
    answer_df = df[['AnswersMid', 'AnswersName']].rename(columns={'AnswersMid': 'MID',
                            'AnswersName': 'Title'})
    
    node_data_df = pd.concat([node_df, answer_df], ignore_index=True).drop_duplicates(subset='MID', keep='first')
    
    node_data_df = node_data_df.sort_values(by='Title', ascending=True)
    
    df.to_csv('./data/freebaseqa_unprocessed.csv', index=False)
    node_data_df.to_csv('./data/node_data_freebaseqa.csv', index=False)
    
    print(f'Number of Questions: {len(question_id)}')
    
    #--------------------------------------------------------------------------
    # valid_df = valid_df[valid_df['InferentialChain'].isin(valid_rels)]

    # # Load the output data from output.csv, assuming it contains columns 'MID' and 'Encoded Title'
    # mid_to_qid = pd.read_csv('./data/mid_qid.csv')

    # entities_map = mid_to_qid[mid_to_qid['MID'].isin(entities)]

    # new_entities = set(entities_map['MID'].tolist())
    # valid_df = df[df['TopicEntityMid'].isin(new_entities) & df['AnswersMid'].isin(new_entities)]
    # valid_question_id = set(valid_df['Question-ID'])
    # not_found = entities - new_entities
