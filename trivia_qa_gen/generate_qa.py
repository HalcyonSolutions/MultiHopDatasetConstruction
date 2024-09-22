# Run from the root directory

from tqdm import tqdm
import pandas as pd
import sys, os, csv

from utils.openai_api import OpenAIHandler
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate trivia questions based on 2-hop paths')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Model to use for generating the questions')
    parser.add_argument('--dataset', type=str, default='./path_quality_scorer/data/multihop/evaluated_2_hop_prune.csv', help='Path to the data file')
    parser.add_argument('--output', type=str, default='./trivia_qa_gen/TriviaQA.txt', help='Path to the output file')
    return parser.parse_args()

def create_prompt(node1, node1_desc, relationship1, node2, node2_desc, relationship2, node3, node3_desc):

    # ! Refine the prompt and only keep one
    # ! or just create a function that generates different prompts 

    # prompt = f"""
    # Trivia Question:

    # Using the following knowledge graph dataset, form a trivia question based on the provided two nodes and relationships that exist between them.
    # The question should incorporate the relationships and lead to the third node as the answer. Question should be one sentence long. Question should contain
    # both node 1 and node 2.

    # Dataset Format:
    # - 2-hop path consisting of 3 nodes and 2 relationships.

    # Input:
    # - Node 1: {node1}   
    # - Node 1 Desciprtion: {node1_desc}    
    # - Relationship 1: {relationship1}    
    # - Node 2: {node2}
    # - Node 2 Description: {node2_desc}    
    # - Relationship 2: {relationship2}
    # - Node 3 Answer: {answer}

    # The question should logically connect Node 1 and Node 2 through their relationships, and answer should not be mentioned in the question.
    # Make sure to construct a question so that the it only has 1 unique answer. Meaning, any other answer would be incorrect.

    # Your response should be a trivia question and answer. In a format: Trivia Question: <question>, Answer: <answer>.
    # """

    ################################################################

    # prompt = f"""
    # Trivia Question:

    # Form a trivia question based on the provided two nodes and their descriptions.
    # The question should incorporate both nodes and lead to the third node as the answer. Question should be one sentence long. Question should contain
    # both node 1 and node 2.

    # Dataset Format:
    # - 2-hop path consisting of 3 nodes.

    # Input:
    # - Node 1: {node1}   
    # - Node 1 Desciprtion: {node1_desc}    
    # - Node 2: {node2}
    # - Node 2 Description: {node2_desc}    
    # - Node 3 Answer: {answer}

    # The question should logically connect Node 1 and Node 2, and answer should not be mentioned in the question.
    # Make sure to construct a question so that it only has 1 unique answer. Meaning, any other similar answer would be incorrect.
    # Example: If the answer is "Apple", the question should not be "What is a fruit?" as it could have multiple answers.

    # Your response should be a trivia question and answer. 
    # In a format: 
    # Trivia Question: <question>
    # Answer: <answer>.
    # """

    ################################################################

    prompt = f"""
    Trivia Question:

    Form a trivia question based on the provided two nodes, and relationships that connect them.
    The question should incorporate both nodes and lead to the third node as the answer. Question should be one sentence long. Question should contain
    both node 1 and node 2 titles.

    Dataset Format:
    - 2-hop path consisting of 3 nodes.

    Input:
    - Node 1 Title: {node1}   
    - Relationship 1: {relationship1}
    - Answer: {node2}
    - Relationship 2: {relationship2}
    - Node 2 Title: {node3}

    The question should logically connect Node 1 and Node 2, and answer should not be mentioned in the question.
    Make sure to construct a question so that it only has 1 unique answer. Meaning, any other similar answer would be incorrect.
    Example: If the answer is "Apple", the question should not be "What is a fruit?" as it could have multiple answers.

    Your response should be a trivia question and answer. 
    In a format: 
    Trivia Question: <question>
    Answer: <answer>.
    """

    ################################################################

    # prompt = f"""
    # Form a sentence incorporating the provided two nodes, their descriptions, and relationships that connect them. The sentence should logically connect Node 1 and Node 2, and should have the answer node masked within it.

    # Dataset Format:
    # - 2-hop path consisting of 3 nodes.

    # Input:
    # - Node 1 Title: {node1}   
    # - Node 1 Description: {node1_desc}
    # - Relationship 1: {relationship1}
    # - Node 2 Title: {node2}
    # - Node 2 Description: {node2_desc}
    # - Relationship 2: {relationship2}
    # - Answer Node: MASK

    # The sentence should incorporate Node 1 and Node 2 and should have the answer node masked with "MASK". Make sure the sentence is coherent and relevant to both nodes. Ensure that the answer node is clearly masked and that the sentence makes sense in the context.

    # Example: If the answer node is "Earth," a possible sentence could be: "I love MASK because I live on it."

    # Your response should be a sentence with the answer node masked. 
    # Format the response as follows: Sentence: <sentence>, Masked Node: <answer>.
    # """

    return prompt

def extract_path(row):
    df_nodes = pd.read_csv('./triplet_creations/data/rdf_data.csv') # helps to find info by RDF
    df_relations = pd.read_csv('./triplet_creations/data/relation_data.csv') # helps to find info by Property 
    path=''
    path = []
    for e in row:
        if e[0] == 'Q':
            node = df_nodes.loc[df_nodes['RDF'] == e]
            node_title = node['Title'].item()
            node_description = node['Description'].item()
            path.append(f'{node_title}')
            path.append(f'{node_description}')
        else:
            relation = df_relations.loc[df_relations['Property'] == e]
            rel_title = relation['Title'].item()
            path.append(f'{rel_title}')

    if path=='':
        raise ValueError('Variable `path` can`t be None!')

    return path

if __name__ == "__main__":
    args = parse_args()
    
    # Load the data
    data = pd.read_csv(args.dataset)
    # evals = column total_score
    evals = data['total_score'].to_list()
    # drop the column total_score, and content
    data = data.drop(columns=['total_score', 'content'])

    # copy the data to a new dataframe
    data_out = data.copy()

    # model
    model = args.model

    # create the OpenAIHandler object
    client = OpenAIHandler(model=model)

    questions = []
    answers = []
    for i in tqdm(range(data.shape[0]), desc="Querying the model"):
        # print(f'=====Data {data.iloc[i]}')
        path = extract_path(data.iloc[i])
        prompt = create_prompt(path[0], path[1], path[2], path[3], path[4], path[5], path[6], path[7])
        response = client.query(prompt)

        for line in response['answer'].split('\n'):
            if line.startswith('Trivia Question:'):
                question = line.split(':')[1].strip()
            if line.startswith('Answer:'):
                answer = line.split(':')[1].strip()

        questions.append(question)
        answers.append(answer)

        # every hundred iterations, save the questions and answers to the output file
        if i % 100 == 0:
            # sicne the question is not long enough extend it to append to the data
            data_out['question'] = questions + [''] * (data.shape[0] - len(questions))
            data_out['answer'] = answers + [''] * (data.shape[0] - len(answers))
            data_out.to_csv(f'datasets/{args.output}', index=False)

    # append two new columns to the data, and save to a new file args.output
    data_out['question'] = questions
    data_out['answer'] = answers
    data_out.to_csv(f'datasets/{args.output}', index=False)
