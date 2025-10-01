"""

"""
import argparse
import random
import pandas as pd
from utils.basic import load_triplets, load_pandas

from typing import Union, List

def parse_args() -> argparse.Namespace:
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Filters FreebaseQA questions to only use those present in the given FB15k Compatible Dataset")
    
    # Input
    parser.add_argument('--multihop-path', type=str, default='./data/paths_kinship_hinton_2hop.txt',
                        help='')
    parser.add_argument('--triplets-path', type=str, default='./data/triplets_kinship_hinton.txt',
                        help='')
    parser.add_argument('--train-split', type=float, default=0.75,
                        help='The proportion of the dataset to include in the train split.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling the dataset.')

    # Output
    parser.add_argument('--kinship-output-path', type=str, default='./data/questions/kinship_hinton_qa_2hop.csv',
                        help='')
    
    return parser.parse_args()

def load_path(file_path: Union[str, List[str]]) -> pd.DataFrame:
    """
    Loads a triplet dataset from one or more file paths into a pandas DataFrame.

    Args:
        file_path (str or list): The path or list of paths to the triplet files.

    Returns:
        pd.DataFrame: A DataFrame containing the triplet data (head, relation, tail).
    """
    if type(file_path) == str:
        # Load the triplets into a DataFrame
        return pd.read_csv(file_path, sep='\t', header=None)
    elif type(file_path) == list:
        # Load and merge the DataFrames from the list of file paths
        df_list = [pd.read_csv(fp, sep='\t', header=None) for fp in file_path]
        return pd.concat(df_list, ignore_index=True)
    else:
        assert False, 'Error! The file_path must either be a string or a list of strings'

def paths_to_triplet(paths: List[List[str]]) -> List[List[List[str]]]:
    """
    Convert paths to triplet format.
    Each path is a list of nodes, and we convert it to a list of triplets.
    Each triplet consists of three consecutive nodes in the path.
    For example, a path [A, B, C, D] would be converted to [[A, B, C], [B, C, D]].
    Args:
        paths (List[List[str]]): A list of paths, where each path is a list of nodes.
    Returns:
        List[List[List[str]]]: A list of triplets, where each triplet is a list of three nodes.

    """
    final_paths = []
    for i0 in range(len(paths)):
        path = paths[i0]
        triplet = []
        i1 = 0
        while i1 < len(path) - 2:
            triplet.append([path[i1], path[i1 + 1], path[i1 + 2]])
            i1 += 2
        final_paths.append(triplet)
    return final_paths

def generate_questions(paths: List[List[str]]) -> List[str]:
    """
    Generate questions based on the paths.
    Each path is a list of triplets, and we generate a question for each triplet.
    Args:
        paths (List[List[str]]): A list of paths, where each path is a list of triplets.
    Returns:
        List[str]: A list of questions generated from the paths.
    """
    questions = []
    questions_alt = []
    source_entities = []
    query_relations = []
    sequential_query_relations = []
    answer_entities = []
    answers = []
    hops = []
    for triplets in paths:
        hops.append(len(triplets))

        
        # reverse the for loop
        question = "Who is "
        for triplet in triplets[::-1]: # Iterate over a reversed copy of the triplet
            question += f"the {triplet[1]} of " # Accessing the relation from the inner list i0
        question += f"{triplet[0]}?"
        questions.append(question)
        source_entities.append(triplet[0]) # Accessing the head from the first triplet

        question_alt = f"Who is {triplet[0]}'s "
        sequential_relations = []
        for triplet in triplets:
            question_alt += f"{triplet[1]}'s "
            sequential_relations.append(triplet[1]) # Accessing the relation from the inner list i0

        question_alt = question_alt[:-3]
        question_alt += "?"
        questions_alt.append(question_alt)

        query_relations.append(triplet[1]) # Accessing the relation from the last 
        sequential_query_relations.append(sequential_relations) # Accessing the relation from the last triplet

        answers.append(triplet[2]) # Accessing the tail from the last triplet
        answer_entities.append(triplet[2]) # Accessing the tail from the last triplet

    return questions, questions_alt, answers, source_entities, query_relations, answer_entities, sequential_query_relations, hops

def eliminate_duplicates(paths: List[List[str]]) -> List[List[str]]:
    """
    Eliminate duplicate paths from the list of paths.
    Args:
        paths (List[List[str]]): A list of paths, where each path is a list of triplets.
    Returns:
        List[List[str]]: A list of unique paths.
    """
    unique_paths = set()
    for path in paths:
        path_tuple = tuple(tuple(triplet) for triplet in path)
        unique_paths.add(path_tuple)
    # convert all tuples of tuples to list of lists
    unique_paths = [list(map(list, path)) for path in unique_paths]

    return unique_paths

def validate_triplet(paths: List[List[str]], triplet_df: pd.DataFrame) -> bool:
    """
    For each triplet in the path, ensure that the head and rel only point to a single tail. If not, do not include the path.
    Args:
        paths (List[List[str]]): A list of paths, where each path is a list of triplets.
        triplet_df (pd.DataFrame): A DataFrame containing the triplet data (head, relation, tail).
    """
    valid_paths = []
    for path in paths:
        valid = True
        for triplet in path:
            head, rel, _ = triplet
            # Check if the head and relation point to a single tail to ensure uniqueness of the triplet
            if len(triplet_df[(triplet_df['head'] == head) & (triplet_df['relation'] == rel)]) != 1:
                valid = False
                break
        if valid:
            valid_paths.append(path)
    return valid_paths

if __name__ == "__main__":
    
    args = parse_args()

    random.seed(args.seed)

    #--------------------------------------------------------------------------
    # Load the paths
    paths_df = load_path(args.multihop_path)
    triplet_df = load_triplets(args.triplets_path)
    
    print(paths_df.head(5))

    # convert df to list
    paths = paths_df.values.tolist()

    # convert paths to triplet format
    paths = paths_to_triplet(paths)
    print(f"Current number of paths: {len(paths)}")

    paths = eliminate_duplicates(paths)
    print(f"Number of unique paths: {len(paths)}")
    print(paths[0:5])

    paths = validate_triplet(paths, triplet_df)
    print(f"Valid number of paths: {len(paths)}")

    questions, question_alt, answers, source_entities, query_relations, answer_entities, sequential_query_relations, hops = generate_questions(paths)
    question_numbers = [i for i in range(len(questions))]
    splitLabels = ['train' if i < len(questions) * args.train_split else 'test' for i in range(len(questions))]
    random.shuffle(splitLabels)

    print(questions[0:5])
    print(question_alt[0:5])

    # Create a DataFrame from the lists
    output_df = pd.DataFrame({
        'Question-Number': question_numbers,            # Question number
        'Question': questions,                          # Original questions
        'Question_Alt': question_alt,                   # Alternative phrasing of the questions
        'Answer': answers,                              # Textual answer
        'Hops': hops,                                   # Number of hops in the path 
        'Source-Entity': source_entities,               # Source entity (starting point of the path)
        'Query-Relation': query_relations,              # Relation between the source and target entities
        'Query-Relations': sequential_query_relations,  # List of relations in the path
        'Answer-Entity': answer_entities,               # Target entity (end point of the path)
        'Paths': paths,                                 # Paths represented as a list of triplets
        'SplitLabel': splitLabels                       # Label indicating if the question is in the training or test set
    })

    print(output_df.head())
    output_df.to_csv(args.kinship_output_path, index=False)