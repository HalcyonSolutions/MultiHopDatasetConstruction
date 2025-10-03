# -*- coding: utf-8 -*-
"""
Knowledge Graph Integrity Checker

Description:
    Validates consistency and integrity of knowledge graph datasets by checking:
    - Consistency between main triplets file and train/test/valid splits
    - Missing entities and relationships in metadata files
    - Data completeness across different components
    
    This tool helps ensure data quality and identifies inconsistencies that could
    affect downstream machine learning tasks or graph analysis.

Usage:
    python kg_integrity_checker.py --triplets-file path/to/triplets.txt --splits-directory path/to/splits/
    
Examples:
    python kg_integrity_checker.py --triplets-file triplets.txt --splits-directory splits/
    python kg_integrity_checker.py --entity-metadata entities.csv --relation-metadata relations.csv
"""

import os
import sys
import argparse
import warnings
from typing import Set, Tuple, Dict

# Add parent directory to path for local imports
sys.path.append('.')

from utils.basic import load_pandas, load_triplets
from utils.wikidata_v2 import update_relationship_data, update_entity_data

# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the integrity checker."""
    parser = argparse.ArgumentParser(
        description="Validates consistency between triplet datasets, train/test/valid splits, and metadata files. "
                   "Checks for missing entities, relationships, and triplets across different data components.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --triplets-file triplets.txt --splits-directory ./splits/
  %(prog)s --update-metadata --entity-metadata entities.csv
        """
    )
    
    # Input file paths
    parser.add_argument('--triplets-file', type=str, 
                       default='./data/link_prediction/Fb-Wiki/triplets.txt',
                       help='Path to main triplets file containing all head-relation-tail triples')
    
    parser.add_argument('--splits-directory', type=str, 
                       default='./data/link_prediction/Fb-Wiki/',
                       help='Directory containing train.txt, valid.txt, and test.txt split files')

    parser.add_argument('--entity-metadata', type=str,
                        default = '',
                    #    default='./data/metadata/node_data_fb_wiki.csv',
                       help='Path to CSV file containing entity metadata (QID, titles, descriptions, etc.). Optional - if not provided, metadata validation will be skipped.')
    
    parser.add_argument('--relation-metadata', type=str, 
                        default = '',
                    #    default='./data/metadata/relation_data_wiki.csv',
                       help='Path to CSV file containing relationship metadata (property IDs, labels, descriptions, etc.). Optional - if not provided, metadata validation will be skipped.')
    
    # Optional flags
    parser.add_argument('--update-metadata', action='store_true',
                       help='Enable updating metadata files with missing entities/relationships found during check')
    
    return parser.parse_args()


# =============================================================================
# Core Functions
# =============================================================================

def load_and_analyze_triplets(triplets_file: str) -> Tuple[object, Set[str], Set[str]]:
    """
    Load main triplets file and extract entity and relationship sets.
    
    Args:
        triplets_file: Path to the triplets file
        
    Returns:
        Tuple of (triplets_dataframe, nodes_set, relationships_set)
    """
    print(f"📁 Loading main triplets from: {triplets_file}")
    triplets_df = load_triplets(triplets_file)
    nodes_set = set(triplets_df['head']) | set(triplets_df['tail'])
    rels_set = set(triplets_df['relation'])
    
    print(f"   ✓ Loaded {len(triplets_df):,} triplets")
    print(f"   ✓ Found {len(nodes_set):,} unique entities")
    print(f"   ✓ Found {len(rels_set):,} unique relationships")
    
    return triplets_df, nodes_set, rels_set

def load_and_analyze_splits(splits_directory: str) -> Tuple[bool, object, Set[str], Set[str]]:
    """
    Load train/test/valid split files and analyze their contents.
    
    Args:
        splits_directory: Directory containing split files
        
    Returns:
        Tuple of (success_flag, splits_dataframe, nodes_set, relationships_set)
    """
    if not (os.path.exists(splits_directory) and os.path.isdir(splits_directory)):
        print(f"⚠️  Splits directory not found: {splits_directory}")
        return False, None, set(), set()
    
    print(f"\n📁 Loading train/valid/test splits from: {splits_directory}")
    
    split_files = ['train.txt', 'valid.txt', 'test.txt']
    split_paths = [os.path.join(splits_directory, fname) for fname in split_files]
    
    # Check if all split files exist
    missing_files = [f for f, path in zip(split_files, split_paths) if not os.path.exists(path)]
    if missing_files:
        print(f"⚠️  Missing split files: {missing_files}")
        return False, None, set(), set()
    
    split_triplets = load_triplets(split_paths)
    split_nodes_set = set(split_triplets['head']) | set(split_triplets['tail'])
    split_rels_set = set(split_triplets['relation'])
    
    print(f"   ✓ Loaded {len(split_triplets):,} triplets from splits")
    print(f"   ✓ Found {len(split_nodes_set):,} unique entities in splits")
    print(f"   ✓ Found {len(split_rels_set):,} unique relationships in splits")
    
    return True, split_triplets, split_nodes_set, split_rels_set


def load_and_analyze_metadata(entity_metadata_path: str, relation_metadata_path: str) -> Tuple[bool, Dict]:
    """
    Load entity and relationship metadata files.
    
    Args:
        entity_metadata_path: Path to entity metadata CSV
        relation_metadata_path: Path to relationship metadata CSV
        
    Returns:
        Tuple of (success_flag, metadata_dict)
    """
    # Check if both paths are provided
    if not entity_metadata_path or not relation_metadata_path:
        print(f"⏭️  Skipping metadata validation (metadata paths not provided)")
        return False, {}
    
    # Check if both files exist
    if not (os.path.exists(entity_metadata_path) and os.path.exists(relation_metadata_path)):
        print(f"⚠️  Metadata files not found:")
        print(f"     Entity: {entity_metadata_path}")
        print(f"     Relation: {relation_metadata_path}")
        print(f"     Skipping metadata validation...")
        return False, {}
    
    print(f"\n📁 Loading metadata files...")
    print(f"   Entity metadata: {entity_metadata_path}")
    print(f"   Relation metadata: {relation_metadata_path}")
    
    node_data_map = load_pandas(entity_metadata_path)
    relation_map = load_pandas(relation_metadata_path)
    
    rel_data_set = set(relation_map['Property'].tolist())
    node_data_set = set(node_data_map['QID'].tolist())
    node_forwarding = set(node_data_map['Forwarding'].tolist())
    
    print(f"   ✓ Loaded {len(node_data_set):,} entities in metadata")
    print(f"   ✓ Loaded {len(rel_data_set):,} relationships in metadata")
    print(f"   ✓ Found {len(node_forwarding):,} forwarding entities")
    
    return True, {
        'node_data_map': node_data_map,
        'relation_map': relation_map,
        'node_data_set': node_data_set,
        'rel_data_set': rel_data_set,
        'node_forwarding': node_forwarding
    }

def validate_splits_consistency(main_triplets_df, main_nodes_set, main_rels_set,
                              split_triplets_df, split_nodes_set, split_rels_set) -> bool:
    """
    Validate consistency between main triplets and split files.
    
    Args:
        main_triplets_df: Main triplets dataframe
        main_nodes_set: Set of nodes from main triplets
        main_rels_set: Set of relationships from main triplets
        split_triplets_df: Split triplets dataframe  
        split_nodes_set: Set of nodes from splits
        split_rels_set: Set of relationships from splits
        
    Returns:
        True if validation passes, False otherwise
    """
    print("\n" + "="*60)
    print("🔍 VALIDATING SPLITS CONSISTENCY")
    print("="*60)
    
    has_errors = False
    
    # Check for missing relationships
    missing_rels = main_rels_set ^ split_rels_set  # symmetric difference
    if missing_rels:
        print(f"❌ Relationship mismatch:")
        print(f"   Main: {len(main_rels_set):,} | Splits: {len(split_rels_set):,}")
        print(f"   Missing relationships: {len(missing_rels):,}")
        print(f"   {missing_rels}")
        has_errors = True
    else:
        print(f"✅ Relationships consistent: {len(main_rels_set):,} relationships")
    
    # Check for missing nodes
    missing_nodes = main_nodes_set ^ split_nodes_set  # symmetric difference
    if missing_nodes:
        print(f"❌ Entity mismatch:")
        print(f"   Main: {len(main_nodes_set):,} | Splits: {len(split_nodes_set):,}")
        print(f"   Missing entities: {len(missing_nodes):,}")
        print(f"   {missing_nodes}")
        has_errors = True
    else:
        print(f"✅ Entities consistent: {len(main_nodes_set):,} entities")
    
    # Check for missing triplets
    main_triplets_tuples = set(map(tuple, main_triplets_df.values))
    split_triplets_tuples = set(map(tuple, split_triplets_df.values))
    missing_triplets = main_triplets_tuples ^ split_triplets_tuples  # symmetric difference
    
    if missing_triplets:
        print(f"❌ Triplet mismatch:")
        print(f"   Main: {len(main_triplets_tuples):,} | Splits: {len(split_triplets_tuples):,}")
        print(f"   Missing triplets: {len(missing_triplets):,}")
        if len(missing_triplets) <= 10:  # Only show if manageable number
            print(f"   {missing_triplets}")
        else:
            print(f"   (Too many to display - {len(missing_triplets):,} missing)")
        has_errors = True
    else:
        print(f"✅ Triplets consistent: {len(main_triplets_tuples):,} triplets")
    
    return not has_errors

def validate_metadata_consistency(main_nodes_set, main_rels_set, metadata_dict, 
                                entity_metadata_path, relation_metadata_path, update_metadata=False) -> bool:
    """
    Validate consistency between main triplets and metadata files.
    
    Args:
        main_nodes_set: Set of nodes from main triplets
        main_rels_set: Set of relationships from main triplets
        metadata_dict: Dictionary containing metadata information
        entity_metadata_path: Path to entity metadata file
        relation_metadata_path: Path to relation metadata file
        update_metadata: Whether to update metadata files with missing entries
        
    Returns:
        True if validation passes, False otherwise
    """
    print("\n" + "="*60)
    print("🔍 VALIDATING METADATA CONSISTENCY")
    print("="*60)
    
    has_errors = False
    
    # Extract metadata sets
    rel_data_set = metadata_dict['rel_data_set']
    node_data_set = metadata_dict['node_data_set']
    node_forwarding = metadata_dict['node_forwarding']
    
    # Check missing relationships in metadata
    missing_rels = main_rels_set - rel_data_set
    if missing_rels:
        print(f"❌ Missing relationships in metadata:")
        print(f"   Triplets: {len(main_rels_set):,} | Metadata: {len(rel_data_set):,}")
        print(f"   Missing: {len(missing_rels):,}")
        print(f"   {missing_rels}")
        
        if update_metadata:
            print("🔄 Updating relationship metadata...")
            rel_df = update_relationship_data(metadata_dict['relation_map'], missing_rels)
            updated_path = relation_metadata_path.replace('.csv', '_updated.csv')
            rel_df.to_csv(updated_path, index=False)
            print(f"   ✓ Updated metadata saved to: {updated_path}")
        
        has_errors = True
    else:
        print(f"✅ Relationships complete in metadata: {len(rel_data_set):,} relationships")
    
    # Check missing entities in metadata
    missing_nodes = main_nodes_set - node_data_set
    if missing_nodes:
        print(f"❌ Missing entities in metadata:")
        print(f"   Triplets: {len(main_nodes_set):,} | Metadata: {len(node_data_set):,}")
        print(f"   Missing: {len(missing_nodes):,}")
        if len(missing_nodes) <= 20:
            print(f"   {missing_nodes}")
        else:
            print(f"   (Too many to display - {len(missing_nodes):,} missing)")
        
        if update_metadata:
            print("🔄 Updating entity metadata...")
            node_df = update_entity_data(metadata_dict['node_data_map'], missing_nodes)
            updated_path = entity_metadata_path.replace('.csv', '_updated.csv')
            node_df.to_csv(updated_path, index=False)
            print(f"   ✓ Updated metadata saved to: {updated_path}")
        
        has_errors = True
    else:
        print(f"✅ Entities complete in metadata: {len(node_data_set):,} entities")
    
    # Check for excess data (warnings only)
    excess_rels = rel_data_set - main_rels_set
    if excess_rels:
        print(f"⚠️  Found {len(excess_rels):,} extra relationships in metadata (this is usually fine)")
    
    excess_nodes = node_data_set - main_nodes_set - node_forwarding
    if excess_nodes:
        print(f"⚠️  Found {len(excess_nodes):,} extra entities in metadata (excluding forwarding)")
    
    return not has_errors


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main function to orchestrate the integrity checking process."""
    print("🚀 Knowledge Graph Integrity Checker")
    print("="*60)
    
    args = parse_args()
    
    # Load and analyze main triplets
    try:
        main_triplets_df, main_nodes_set, main_rels_set = load_and_analyze_triplets(args.triplets_file)
    except Exception as e:
        print(f"❌ Error loading main triplets: {e}")
        return False
    
    # Load and analyze splits
    evaluate_splits, split_triplets_df, split_nodes_set, split_rels_set = load_and_analyze_splits(args.splits_directory)
    
    # Load and analyze metadata
    evaluate_metadata, metadata_dict = load_and_analyze_metadata(args.entity_metadata, args.relation_metadata)
    
    # Ensure at least one validation can be performed
    if not evaluate_splits and not evaluate_metadata:
        print("\n❌ ERROR: No valid split directory found and no metadata provided!")
        print("   Please check the splits directory path or provide metadata file paths.")
        return False
    
    # Perform validations
    validation_results = []
    
    if evaluate_splits:
        splits_valid = validate_splits_consistency(
            main_triplets_df, main_nodes_set, main_rels_set,
            split_triplets_df, split_nodes_set, split_rels_set
        )
        validation_results.append(("Splits", splits_valid))
    
    if evaluate_metadata:
        metadata_valid = validate_metadata_consistency(
            main_nodes_set, main_rels_set, metadata_dict,
            args.entity_metadata, args.relation_metadata, args.update_metadata
        )
        validation_results.append(("Metadata", metadata_valid))
    
    # Final results
    print("\n" + "="*60)
    print("📊 FINAL VALIDATION RESULTS")
    print("="*60)
    
    all_valid = True
    for check_name, is_valid in validation_results:
        status = "✅ PASSED" if is_valid else "❌ FAILED"
        print(f"{check_name:20} {status}")
        all_valid &= is_valid
    
    print("-"*60)
    if all_valid:
        print("🎉 Overall Status: ALL CHECKS PASSED")
        print("   Your knowledge graph data is consistent!")
    else:
        print("💥 Overall Status: SOME CHECKS FAILED")
        print("   Please review the errors above and fix the issues.")
        if not args.update_metadata:
            print("   Consider using --update-metadata to automatically fix metadata issues.")
    
    return all_valid


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)