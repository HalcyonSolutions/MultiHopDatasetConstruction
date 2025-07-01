from typing import Set, Tuple
import json
from tqdm import tqdm

def extract_mquake_entities(
    mquake_path: str,
) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    """
    Extract WikiData entity and relation IDs from MQuAKE questions and answers.
    
    Args:
        mquake_path: Path to the MQuAKE dataset file (standard JSON)
    
    Returns:
        Tuple containing:
        - Set of entity IDs (Q-prefixed) extracted from MQuAKE
        - Set of relation IDs (P-prefixed) extracted from MQuAKE
        - Set of entity IDs (Q-prefixed) extracted from MQuAKE counterfactual set
        - Set of relation IDs (P-prefixed) extracted from MQuAKE counterfactual set
    """
    entities = set()  # For Q-prefixed entity IDs
    relations = set()  # For P-prefixed relation IDs

    rr_entities = set()
    rr_relations = set()

    # Process the main MQuAKE file

    with open(mquake_path, 'r') as f:
        # Load the entire JSON file (array of examples)
        mquake_data = json.load(f)

    # Extract entity IDs from requested_rewrite
    for i, example in tqdm(enumerate(mquake_data), desc="Extracting data"):
        # TEST: Assumption to test if there are rewrites and orig triplets on each examplet
        assert (
            "requested_rewrite" in example
            and "orig" in example
            and "triples" in example["orig"]
        ), "Not every sample provided contains request_rewrite and original triplets"

        for h,r,t in example["orig"]["triples"]:
            entities.add(h)
            relations.add(r)
            entities.add(t)

        for rewrite in example["requested_rewrite"]:
            # Extract relation ID (Wikidata Property)
            if "relation_id" in rewrite and rewrite["relation_id"].startswith("P"):
                rr_relations.add(rewrite["relation_id"])

            # Extract target entity IDs
            if "target_new" in rewrite and "id" in rewrite["target_new"] and rewrite["target_new"]["id"].startswith("Q"):
                rr_entities.add(rewrite["target_new"]["id"])

    return entities, relations, rr_entities, rr_relations

def extract_mquake_triplets(mquake_path: str) -> Set[Tuple[str, str, str]]:
    """
    Extract pure triplets from MQuAKE dataset.

    Args:
        mquake_path: Path to the MQuAKE dataset file (standard JSON)

    Returns:
        Set of triplets (head, relation, tail) extracted from MQuAKE
    """
    triplets = set()

    with open(mquake_path, 'r') as f:
        mquake_data = json.load(f)

    for i, example in tqdm(enumerate(mquake_data), desc="Extracting triplets"):
        assert (
            "orig" in example
            and "triples" in example["orig"]
        ), "Not every sample provided contains original triplets"

        for h, r, t in example["orig"]["triples"]:
            triplets.add((h, r, t))

    return triplets
