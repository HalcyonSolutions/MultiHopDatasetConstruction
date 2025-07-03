#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script expands the MQuAKE triplet dataset by querying WikiData to create a larger
knowledge graph dataset comparable to FB15k in size. It handles entity extraction from MQuAKE,
expands entity relationships through WikiData API calls, and processes the resulting triplets
to create a high-quality dataset for knowledge graph embedding tasks.

Key functionalities:
1. **Entity & Relation Extraction**: Extracts entities and relations from MQuAKE dataset
2. **Neighborhood Expansion**: Expands the entity set through N-hop neighbors
3. **WikiData Querying**: Uses WikiData API to retrieve all relationships for entities
4. **Triplet Processing**: Handles filtering, cleaning, and deduplication
5. **Dataset Creation**: Splits into train/test/validation for knowledge graph embedding

This pipeline can be run end-to-end or in stages:
- **Entity Extraction**: Extract initial entities from MQuAKE
- **Entity Expansion**: Expand the entity set through WikiData neighbors
- **Triplet Generation**: Query WikiData for all relationships within the entity set
- **Dataset Creation**: Create the final processed dataset
- **Misc**: Additional tools to run statistics or benchmarking

Usage:
    python mquake_triplet_process.py 
    python mquake_triplet_process.py --mode extract_entities --mquake_path [MQuAKE-CF.json path]
    python mquake_triplet_process.py --mode expand_entities
    python mquake_triplet_process.py --mode create_datasets
    python mquake_triplet_process.py --mode full_pipeline 
    python mquake_triplet_process.py --mode convert_triplets_for_stats 
"""

import argparse
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.common import StrTriplet
from utils.logging import create_logger
from utils.process_triplets import get_relations_and_entities_to_prune
from utils.wikidata_v2 import fetch_entity_triplet, process_entity_triplets, retry_fetch
from utils.mquake import extract_mquake_entities


ETWQ_COLUMNS = ["head", "rel", "tail", "qualifiers"]


def parse_args():
    """Parse command line arguments for the MQuAKE triplet expansion pipeline."""
    parser = argparse.ArgumentParser(description="Expand MQuAKE triplets by querying WikiData")

    # Pipeline mode selection
    parser.add_argument("--mode", type=str, 
                        choices=["extract_entities", "expand_entities","convert_triplets_for_stats", "create_dataset", "full_pipeline", "multihop_entities_relations", "pruning_expanded_triplets"],
                        default="full_pipeline", help="Operating mode for the pipeline")

    # MQuAKE input paths
    parser.add_argument("--mquake_path", type=str, default="./data/MQuAKE-CF.json",
                        help="Path to the MQuAKE dataset file")

    ## FilePaths
    # Pre processing
    parser.add_argument("--outPath_og_entities", type=str, default="./data/mquake/entities.txt",
                        help="Path to save the extracted entities")
    parser.add_argument("--outPath_og_relation", type=str, default="./data/mquake/relations.txt",
                        help="Path to save the extracted relations")
    parser.add_argument("--outPath_cf_entities", type=str, default="./data/mquake/cf_entities.txt",
                        help="Path to save the extracted entities")
    parser.add_argument("--outPath_cf_relations", type=str, default="./data/mquake/cf_relations.txt",
                        help="Path to save the extracted relations")
    parser.add_argument("--outPath_processedTriplets", type=str, default="./data/mquake/processed_triplets.txt",
                        help="Path to save processed triplets")
    # Post-processing
    parser.add_argument("--outPath_expanded_triplets", type=str, default="./data/mquake/expanded_triplets.csv",
                        help="Path to save the expanded triplet set")
    parser.add_argument("--outPath_expanded_triplet_wo_qualifiers", type=str, default="./data/mquake/expanded_triplets_wo_qualifiers.csv",
                        help="Path to save the expanded triplet set")
    parser.add_argument("--outPath_expanded_entity_set", type=str, default="./data/mquake/expanded_entities.txt",
                        help="Path to save the expanded entity set")
    parser.add_argument("--outPath_expanded_relation_set", type=str, default="./data/mquake/expanded_relations.txt",
                        help="Path to save the expanded relation set")
    parser.add_argument("--outPath_expNPruned_ents", type=str, default="./data/mquake/expandedNpruned_entities.txt",
                        help="Path to save the expanded and pruned entities from counterfactual set")
    parser.add_argument("--outPath_expNPruned_rels", type=str, default="./data/mquake/expandedNpruned_relations.txt",
                        help="Path to save the expanded and pruned relations from counterfactual set")
    parser.add_argument( "--outPath_expNPruned_triplets", default="./data/mquake/expNpruned_triplets.txt",
        type=str, help="Location to save pruned triplets to to",)
    # Dataset creation 
    parser.add_argument("--outPath_train_split", type=str, default="./data/mquake/train.txt",
                        help="Path to save the training triplets")
    parser.add_argument("--outPath_test_split", type=str, default="./data/mquake/test.txt",
                        help="Path to save the test triplets")
    parser.add_argument("--outPath_valid_split", type=str, default="./data/mquake/valid.txt",
                        help="Path to save the validation triplets")

    ## Helper data
    parser.add_argument("--relationship_hierarchy_mapping_path", type=str, default="./data/relationships_hierarchy.txt",
                        help="Path to the file containing relationship hierarchies for processing.")

    # Triplet processing parameters
    parser.add_argument("--remove_inverse_relationships", action="store_false", help="Whether to remove inverse relationships")
    parser.add_argument("--enable_bidirectional_removal", action="store_false", help="Whether to remove bidirectional relationships")

    # Entity expansion parameters
    parser.add_argument("--expansion_hops", type=int, default=2,
                        help="Number of hops to expand from core entities")
    parser.add_argument("--target_entity_count", type=int, default=15000,
                        help="Target number of entities to include in the dataset")
    parser.add_argument("--entity_batch_size", type=int, default=512,
                        help="Number of entities to process in each batch for expansion")
    parser.add_argument("--use_qualifiers_for_expansion", action="store_true", help="Whether to use qualifiers for entity expansion")

    # Filtering 
    parser.add_argument("--ent_rel_pruning_threshold", type=int, default=40,
        help="Number of appearances in triplets necessary for an entity and realtions to appear.")

    # Triplet generation parameters
    parser.add_argument("--max_workers", type=int, default=5,
                        help="Maximum number of concurrent workers for WikiData API queries. 5 is the limit set by wikimedia foundation.")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Maximum number of retries for failed WikiData queries")
    parser.add_argument("--timeout", type=int, default=5,
                        help="Timeout in seconds for WikiData queries")
    parser.add_argument("--target_triplet_count", type=int, default=200000,
                        help="Target number of triplets for the final dataset")

    # Dataset split parameters
    parser.add_argument("--split_dataset")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Proportion of triplets to use for training")
    parser.add_argument("--test_valid_ratio", type=float, default=0.5,
                        help="Proportion of non-training triplets to use for testing vs validation")

    # Additional options
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    return parser.parse_args()


def _batch_entity_set_expansion(
    batch: list[str],
    max_workers: int,
    max_retries: int,
    timeout: int,
    use_qualifiers_for_expansion: bool,
) -> Tuple[Set[StrTriplet], Dict[str,str],Dict]:
    """
    Batch-wise entity expansion

    Returns (2)
    ---------
    - newfound_triplets: Set of new triplets
    - forward_dict: Dictionary detailing updates in id as dicated by wikidata
    - qualifier_dictionary: Dictionary of qualifier triplets
    """

    newfound_triplets: Set[StrTriplet] = set()
    qualifier_dictionary = dict()
    forward_dict: dict[str,str] = dict()
    # Use ThreadPoolExecutor to fetch neighbors concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        # Submit tasks to fetch neighbors for each entity in the batch
        futures = {
            executor.submit(
                retry_fetch,
                fetch_entity_triplet,
                entity,
                "expanded" if use_qualifiers_for_expansion else "separate",
                max_retries=max_retries,
                timeout=timeout,
                verbose=True,
            ): entity
            for entity in batch
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            entity = futures[future]
            try:
                _newfound_triplets, _forward_dict, _qualifier_dict  = future.result(timeout=timeout)

                newfound_triplets.update(_newfound_triplets)
                qualifier_dictionary.update(_qualifier_dict)
                forward_dict.update(_forward_dict)

            except Exception as e:
                logger.info(f"Error processing entity {entity}: \n\t{e}")
                # Unrecoverable error
                exit(-1)

    return newfound_triplets, forward_dict, qualifier_dictionary


def expand_triplet_set(
    entity_set: Set[str],
    target_size: int,
    expansion_hops: int,
    use_qualifiers_for_expansion: bool,
    batch_size: int,
    max_workers: int, 
    max_retries: int,
    timeout: int,
) -> tuple[set[tuple[str, str, str]], dict[tuple,Any]]:
    """
    Expand entity and relation set by finding neighbors of existing entities through WikiData.
    And then create new triplets from the expanded entity set.
    
    Args:
        entity_set: Initial set of entities to expand from
        target_size: Target number of entities in the expanded set
        expansion_hops: Number of hops to expand (careful with values > 1)
        batch_size: Number of entities to process in each batch
        max_workers: Maximum number of concurrent workers for WikiData API queries
        max_retries: Maximum number of retries for failed WikiData queries
        timeout: Timeout in seconds for WikiData queries
    
    Returns:
        Expanded set of entities
    """
    logger.info(f"Target size: {target_size}, Expansion hops: {expansion_hops}")
    
    # Initialize the expanded entity set with the initial entities
    expanded_triplets: Set[StrTriplet] = set()
    forward_dict: dict[str, str] = dict()
    qualifier_dictionary = dict()
    entities_to_process_per_hop = list(entity_set)
    processed_entities = set()
    
    num_ogEntities_processed = 0
    
    # Process in hops
    for hop in range(expansion_hops):
            
        logger.info(f"Beginning hop {hop+1} with {len(entities_to_process_per_hop)} entities to process")
        
        # Track neighbors found in this hop
        new_neighbors = set()
        # Process in batches
        batch_num = 0
        # _exit = False # DEBUG: Tool
        batch_steps = range(0, len(entities_to_process_per_hop), batch_size)
        for batch_start in tqdm(batch_steps, desc="Processing Entities"):

            batch_end = min(batch_start + batch_size, len(entities_to_process_per_hop))
            batch = entities_to_process_per_hop[batch_start:batch_end]
            batch = [entity for entity in batch if entity not in processed_entities]

            _expanded_triplets, _forward_dict, _qualifier_dictionary = _batch_entity_set_expansion(
                batch = batch,
                max_workers = max_workers,
                max_retries = max_retries,
                timeout = timeout,
                use_qualifiers_for_expansion=use_qualifiers_for_expansion,
            )
            logger.debug(f"At {batch_num} we have process {len(_expanded_triplets)} triplets")
            num_ogEntities_processed += batch_size
            logger.debug(f"At {batch_num} we have process {num_ogEntities_processed}")

            processed_entities.update(batch)

            expanded_triplets.update(_expanded_triplets)
            forward_dict.update(_forward_dict)
            qualifier_dictionary.update(_qualifier_dictionary)

            # Use tails in newfound_triplets to expand the entity set
            new_tails = [tails for _, _, tails in _expanded_triplets]
            new_neighbors.update(new_tails)
        
        # If no new neighbors were found, we can't expand further
        if not new_neighbors:
            logger.info("No new neighbors found, stopping expansion")
            break
            
        # Prepare for next hop if needed
        entities_to_process_per_hop = list(new_neighbors - processed_entities)
        logger.info(f"Hop {hop+1} complete. Found {len(new_neighbors)} new neighbors.")
        logger.info(f"Total entities now: {len(expanded_triplets)}")
    
    logger.info(f"Entity expansion complete. Final count: {len(expanded_triplets)}")
    
    return expanded_triplets, qualifier_dictionary


def generate_triplets_from_entities(
    entity_file: str,
    output_file: str,
    max_workers: int = 20,
    max_retries: int = 3,
    timeout: int = 5,
    target_triplet_count: int = 200000,
) -> None:
    """
    Generate triplets by querying WikiData for relationships between entities.
    
    Args:
        entity_file: Path to the file containing entities
        output_file: Path to save the generated triplets
        max_workers: Maximum number of concurrent workers for WikiData API queries
        max_retries: Maximum number of retries for failed WikiData queries
        timeout: Timeout in seconds for WikiData queries
        target_triplet_count: Target number of triplets to generate
    """
    logger.info(f"Generating triplets from entities in {entity_file}")
    logger.info(f"Target triplet count: {target_triplet_count}")
    
    # Use the existing process_entity_triplets function from utils.wikidata_v2
    process_entity_triplets(
        entity_file,
        output_file,
        max_workers=max_workers,
        max_retries=max_retries,
        timeout=timeout,
    )
    
    # Count the generated triplets
    triplet_count = 0
    with open(output_file, 'r') as f:
        for _ in f:
            triplet_count += 1
    
    logger.info(f"Generated {triplet_count} triplets")
    
    if triplet_count < target_triplet_count:
        logger.info(f"Warning: Generated triplet count ({triplet_count}) is less than target ({target_triplet_count})")
        logger.info("Consider expanding the entity set or increasing the expansion hops")


def create_dataset_splits(
    triplets: List[StrTriplet],
    train_file: str,
    test_file: str,
    valid_file: str,
    train_ratio: float = 0.8,
    test_valid_ratio: float = 0.5,
    seed: int = 42,
) -> None:
    """
    Split processed triplets into train/test/validation sets for model training.
    
    Args:
        processed_triplet_file (List[tuple[str,str,str]]): 
        train_file: Path to save the training triplets
        test_file: Path to save the test triplets
        valid_file: Path to save the validation triplets
        train_ratio: Proportion of triplets to use for training
        test_valid_ratio: Proportion of non-training triplets to use for testing vs validation
        seed: Random seed for reproducibility
    """
    logger.info(f"Triplets look a bit like this: {triplets[0]}")
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Count total triplets
    total_triplets = len(triplets)
    
    logger.info(f"Loaded {total_triplets} triplets")
    logger.info(f"Train ratio: {train_ratio}, Test-valid ratio: {test_valid_ratio}")
    
    # Track entity and relation statistics
    entity_to_triplets = defaultdict(list)
    relation_to_triplets = defaultdict(list)
    
    for i, (head, rel, tail) in enumerate(triplets):
        entity_to_triplets[head].append(i)
        entity_to_triplets[tail].append(i)
        relation_to_triplets[rel].append(i)
    
    # Determine split sizes
    train_size = int(total_triplets * train_ratio)
    test_valid_size = total_triplets - train_size
    test_size = int(test_valid_size * test_valid_ratio)
    valid_size = test_valid_size - test_size
    
    logger.info(f"Projected train size: {train_size}, test size: {test_size}, valid size: {valid_size}")
    
    # Initialize sets for tracking
    train_indices = set()
    test_indices = set()
    valid_indices = set()
    
    # First, ensure each relation has examples in each split
    min_examples_per_split = 3
    
    for rel, triplet_indices in relation_to_triplets.items():
        if len(triplet_indices) < min_examples_per_split * 3:
            # Too few examples, just put all in training
            train_indices.update(triplet_indices)
        else:
            # Shuffle and allocate
            shuffled = triplet_indices.copy()
            random.shuffle(shuffled)
            
            # Add some to each split
            valid_indices.update(shuffled[:min_examples_per_split])
            test_indices.update(shuffled[min_examples_per_split:min_examples_per_split*2])
            train_indices.update(shuffled[min_examples_per_split*2:])
    
    # Fill remaining slots
    remaining_indices = set(range(total_triplets)) - train_indices - test_indices - valid_indices
    remaining_list = list(remaining_indices)
    random.shuffle(remaining_list)
    
    # Calculate how many more we need for each split
    train_needed = train_size - len(train_indices)
    test_needed = test_size - len(test_indices)
    valid_needed = valid_size - len(valid_indices)
    
    # Add remaining to appropriate splits
    if train_needed > 0:
        train_indices.update(remaining_list[:train_needed])
        remaining_list = remaining_list[train_needed:]
    
    if test_needed > 0 and remaining_list:
        test_indices.update(remaining_list[:test_needed])
        remaining_list = remaining_list[test_needed:]
    
    if valid_needed > 0 and remaining_list:
        valid_indices.update(remaining_list[:valid_needed])
    
    # Prepare final triplet lists
    train_triplets = [triplets[i] for i in train_indices]
    test_triplets = [triplets[i] for i in test_indices]
    valid_triplets = [triplets[i] for i in valid_indices]
    
    # Save splits to files
    with open(train_file, 'w') as f:
        for head, rel, tail in train_triplets:
            f.write(f"{head} {rel} {tail}\n")
    
    with open(test_file, 'w') as f:
        for head, rel, tail in test_triplets:
            f.write(f"{head} {rel} {tail}\n")
    
    with open(valid_file, 'w') as f:
        for head, rel, tail in valid_triplets:
            f.write(f"{head} {rel} {tail}\n")

    logger.info("Dataset split complete:")
    logger.info(f"- Training: {len(train_triplets)} triplets ({len(train_triplets)/total_triplets*100:.1f}%)")
    logger.info(f"- Testing: {len(test_triplets)} triplets ({len(test_triplets)/total_triplets*100:.1f}%)")
    logger.info(f"- Validation: {len(valid_triplets)} triplets ({len(valid_triplets)/total_triplets*100:.1f}%)")

def filtering_triplets():
    """
    Will filter through triplets ensuring only non-sparse relations remain.
    Additionally, it will
    """
    pass

def expand_triplets_logistics(
    path_og_entities: str,
    path_triplets_w_qualifier: str,
    path_expanded_entities: str,
    path_expanded_relations: str,
    target_entity_count: int,
    expansion_hops: int,
    use_qualifiers_for_expansion: bool,
    entity_batch_size: int,
    max_workers: int,
    max_retries: int,
    timeout: int,
) -> None:

    # By default, will just used origina entities.
    with open(path_og_entities, 'r') as f: 
        entities_to_expand_upon = {line.strip() for line in f}

    logger.info(f"Expanding the original entity set from {len(entities_to_expand_upon)} initial entities...")

    ########################################
    # Core Logic of Expansion 
    ########################################
    expanded_triplets, qualifier_dict = expand_triplet_set(
        entity_set=entities_to_expand_upon,
        target_size=target_entity_count,
        expansion_hops=expansion_hops,
        use_qualifiers_for_expansion=use_qualifiers_for_expansion,
        batch_size=entity_batch_size,
        max_workers=max_workers,
        max_retries=max_retries,
        timeout=timeout,
    )
    ########################################
    # Final Save of Data 
    ########################################
    expanded_triplets_w_qualifiers = []
    expanded_entities = set()
    expanded_relations = set()
    for new_triplet in expanded_triplets:
        if new_triplet in qualifier_dict:
            expanded_triplets_w_qualifiers.append((*new_triplet, str(qualifier_dict[new_triplet])))
        else:
            expanded_triplets_w_qualifiers.append((*new_triplet, ""))

        # Expand the entities and relations
        expanded_entities.update((new_triplet[0], new_triplet[2]))
        expanded_relations.add(new_triplet[1])

    expanded_triplets_w_qualifiers_df = pd.DataFrame(expanded_triplets_w_qualifiers, columns=ETWQ_COLUMNS) # type: ignore
    expanded_triplets_w_qualifiers_df.to_csv(path_triplets_w_qualifier, index=False)
    logger.info(f"Saved {len(expanded_triplets)} expanded triplets to {path_triplets_w_qualifier}")

    # Dump the expanded entities and relations
    with open(path_expanded_entities, 'w') as f:
        for entity in sorted(expanded_entities):
            f.write(f"{entity}\n")
    with open(path_expanded_relations, 'w') as f:
        for relation in sorted(expanded_relations):
            f.write(f"{relation}\n")
    logger.info(f"Saved {len(expanded_entities)} expanded entities to {path_expanded_entities}")
    logger.info(f"Saved {len(expanded_relations)} expanded relations to {path_expanded_relations}")


def main():
    """Main function to execute the MQuAKE triplet expansion pipeline."""
    args = parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    paths_to_create = [
        args.outPath_og_entities,
        args.outPath_og_relation,
        args.outPath_cf_entities,
        args.outPath_cf_relations,
        args.outPath_expanded_triplets,
        args.outPath_processedTriplets,
        args.outPath_train_split,
        args.outPath_test_split,
        args.outPath_valid_split,
    ]

    # Create output directories if they don't exist
    for path in paths_to_create:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    ########################################
    # Full Pipeline
    ########################################

    # Execute pipeline based on mode
    if args.mode in ['extract_entities', 'full_pipeline']:
        # Extract entities from MQuAKE data
        og_entities, og_relations, cf_entities, cf_relations = extract_mquake_entities(
            args.mquake_path, 
        )
        paths_to_save = [
            args.outPath_og_entities,
            args.outPath_og_relation,
            args.outPath_cf_relations,
            args.outPath_cf_relations,
        ]
        entities_to_save = [og_entities, og_relations, cf_entities, cf_relations]

        for path, ents in zip(paths_to_save, entities_to_save):
            with open(path, 'w') as f:
                for entity in sorted(ents):
                    f.write(f"{entity}\n")
            logger.info(f"Saved {len(ents)} RDFs to {path}")
        logger.info("Note: At the moment the counter factual entities and relations have not use. It is in case they are needed in the future.")

    if args.mode in ['expand_entities', 'full_pipeline']:
        # Load MQuAKE entities if not already extracted in this run
        expand_triplets_logistics(
            path_og_entities = args.outPath_og_entities,
            path_expanded_entities= args.outPath_expanded_entity_set, 
            path_triplets_w_qualifier = args.outPath_expanded_triplets,
            path_expanded_relations= args.outPath_expanded_relation_set,
            target_entity_count = args.target_entity_count,
            expansion_hops = args.expansion_hops,
            use_qualifiers_for_expansion = args.use_qualifiers_for_expansion,
            entity_batch_size = args.entity_batch_size,
            max_workers = args.max_workers,
            max_retries = args.max_retries,
            timeout = args.timeout 
        )

    ########################################
    # Post Processing Modes; Mostly for utility:
    ########################################
    if args.mode in ["convert_triplets_for_stats", "full_pipeline"]:
        # Helpfpul for converting the triplets to a format for statistics
        expanded_triplets_w_qualifiers_df = pd.read_csv(
            args.outPath_expanded_triplets,
            header=None,
            names=["head", "relation", "tail", "qualifiers"],
        )
        expanded_triplets_wo_qualifiers = expanded_triplets_w_qualifiers_df.loc[:, ["head", "relation", "tail"]]
        expanded_triplets_wo_qualifiers.to_csv(args.outPath_expanded_triplet_wo_qualifiers, index=False, sep="\t")
        logger.info(f"Removed qualifiers from {args.outPath_expanded_triplets} and saved it to {args.outPath_expanded_triplet_wo_qualifiers}")

    if args.mode in ["pruning_expanded_triplets"]:
        # Load Entities and Relations that canot be pruned
        non_prunable_entities: Set[str] = set()
        non_prunable_relations: Set[str] = set()
        with open(args.outPath_og_entities, 'r') as f:
            for line in f:
                non_prunable_entities.add(line.strip())
        with open(args.outPath_og_relation, 'r') as f:
            for line in f:
                non_prunable_relations.add(line.strip())
        logger.info(f"Loaded {len(non_prunable_entities)} non prunable entities and"
                    f"Loaded {len(non_prunable_relations)} non prunable relations")

        # Load the expanded triplets
        logger.info(f"Loading expanded triplets from {args.outPath_expanded_triplet_wo_qualifiers}")
        expanded_triplets_df = pd.read_csv(args.outPath_expanded_triplet_wo_qualifiers, sep="\t", header=None, names=["head", "relation", "tail"])
        logger.info(f"Expanded (of length: {len(expanded_triplets_df)} triplets: {expanded_triplets_df.head()}")

        # Prune the triplets
        ents_to_prune, rels_to_prune = get_relations_and_entities_to_prune(
            triplets_df=expanded_triplets_df,
            non_prunable_entities=set(non_prunable_entities),
            non_prunable_relations=set(non_prunable_relations),
            pruning_upper_thresh=args.ent_rel_pruning_threshold,
        )
        ents_to_prune = list(ents_to_prune)
        rels_to_prune = list(rels_to_prune)
        logger.info(f"About to prune {len(ents_to_prune)} entities and {len(rels_to_prune)} relations")

        # Filter out pruned entities and relations
        logger.info(f"expanded_triplets_df now contains {len(expanded_triplets_df)} triplets")
        pruned_triplets_df = expanded_triplets_df[
            ~expanded_triplets_df['head'].isin(ents_to_prune) &
            ~expanded_triplets_df['tail'].isin(ents_to_prune) &
            ~expanded_triplets_df['relation'].isin(rels_to_prune)
        ]
        logger.info(f"{len(pruned_triplets_df)} triplets remain after pruning")

        # Calculate new Entity and Relation Set
        ent_set = pd.concat([pruned_triplets_df['head'], pruned_triplets_df['tail']]).drop_duplicates()
        rel_set = pruned_triplets_df['relation'].drop_duplicates()

        # Save the new Entity and Relation Set
        ent_set.to_csv(args.outPath_expNPruned_ents, index=False, sep="\t", header=False)
        rel_set.to_csv(args.outPath_expNPruned_rels, index=False, sep="\t", header=False)
        logger.info(f"Saved {len(rel_set)} expanded and pruned relations to {args.outPath_expNPruned_rels}")
        logger.info(f"Saved {len(ent_set)} expanded and pruned entities to {args.outPath_expNPruned_ents}")

        logger.info(f"Total amount of Pruned {len(pruned_triplets_df)} triplets")
        logger.info(f"Saved {len(pruned_triplets_df)} expanded and pruned triplets to {args.outPath_expNPruned_triplets}")
        pruned_triplets_df.to_csv(args.outPath_expNPruned_triplets, index=False, sep="\t")

    if args.mode in ["create_dataset"]:
        # Create dataset splits

        triplets: List[StrTriplet] = []
        # Ensure that the expanded triplets file exists
        if not os.path.exists(args.outPath_expNPruned_triplets):
            logger.error(
                f"Expanded triplets file {args.outPath_expanded_triplet_wo_qualifiers} does not exist."
                " Please run the pipeline first.A"
                " And ensure to run `convert_triplets_for_stats` mode before `create_dataset` mode."
            )
            exit(-1)

        # Load the expanded triplets file
        with open(args.outPath_expNPruned_triplets, 'r') as f:
            for line_no, line in enumerate(f):
                if line_no == 0:
                    continue
                splits = line.strip().split("\t")
                assert len(splits) == 3
                triplets.append((splits[0], splits[1], splits[2]))

        create_dataset_splits(
            triplets,
            args.outPath_train_split,
            args.outPath_test_split,
            args.outPath_valid_split,
            train_ratio=args.train_ratio,
            test_valid_ratio=args.test_valid_ratio,
            seed=args.seed,
        )

    if args.mode in ["multihop_entities_relations"]: 
        # Then we just add some ids at the begining to make it compatible to MultiHopKG graph embedding training
        entitiy_counter = 0
        with open(args.outPath_expNPruned_ents, 'r') as f:
            entities = f.readlines()
        dirname = os.path.dirname(args.outPath_expanded_entity_set)

        compliant_entities_file_name = os.path.join(dirname, "entities.dict")
        with open(compliant_entities_file_name, 'w') as f:
            for line in entities:
                f.write(f"{entitiy_counter}\t{line.strip()}\n")
                entitiy_counter += 1
        logger.info(f"Saved entities dictionary into {compliant_entities_file_name}")

        relation_counter = 0
        with open(args.outPath_expNPruned_rels, 'r') as f:
            relations = f.readlines()
        compliant_relations_file_name = os.path.join(dirname, "relations.dict")
        with open(compliant_relations_file_name, 'w') as f:
            for line in relations:
                f.write(f"{relation_counter}\t{line.strip()}\n")
                relation_counter += 1
        logger.info(f"Saved relations dictionary into {compliant_entities_file_name}")


    #########################################
    # Final Summary
    ########################################
    if args.mode == "full_pipeline":
        logger.info("\nFull pipeline completed successfully!")
        logger.info(f"- Initial MQuAKE entities: {args.outPath_og_entities}")
        logger.info(f"- Initial MQuAKE Relations: {args.outPath_og_relation}")
        logger.info(f"- Expanded Entity set: {args.outPath_expanded_entity_set}")
        logger.info(f"- Expanded Relations set: {args.outPath_expanded_relation_set}")
        logger.info(f"- Expanded Processed triplets (w/ qualifiers): {args.outPath_expanded_triplets}")



if __name__ == "__main__":
    logger = create_logger("mquake_triplet_process")
    main() 
