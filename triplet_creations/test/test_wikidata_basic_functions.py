
# -*- coding: utf-8 -*-
"""
Wikidata Basic Functions Test Suite

Description:
    This script tests and verifies the core functionality of the local Wikidata utility package.
    It performs comprehensive tests on fundamental Wikidata operations including:
    - Client initialization and configuration
    - Entity search by name
    - Entity details retrieval
    - Triplet extraction for entities and properties
    - Error handling and data validation

    This is part of a larger test suite for Wikidata utilities. Additional specialized tests
    can be found in other test files for specific functionality areas.

Usage:
    python test_wikidata_basic_functions.py
    
This serves as both a foundational test suite and demonstration of core Wikidata utilities.
"""

import sys
import os
from typing import List, Dict, Any

# Add parent directory to path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from utils.wikidata_v2 import (
        get_thread_local_client, 
        search_wikidata_relevant_id,
        fetch_entity_details, 
        fetch_head_entity_triplets,
        fetch_relationship_details, 
        fetch_relationship_triplet
    )
    print("✅ Successfully imported Wikidata utility functions")
except ImportError as e:
    print(f"❌ Error importing Wikidata utilities: {e}")
    print("   Please ensure the utils package is available in the parent directory")
    sys.exit(1)


# =============================================================================
# Test Configuration
# =============================================================================

class TestConfig:
    """Configuration for Wikidata integrity tests."""
    
    # Test queries
    QUERY_TEXT = "Barack Obama"
    QUERY_ENTITY = "Q76"  # Barack Obama
    QUERY_PROPERTY = "P39"  # Position Held
    
    # Limits
    SEARCH_LIMIT = 3
    TRIPLET_LIMIT = 10
    
    # Display settings
    SEPARATOR = "=" * 60
    SUBSEPARATOR = "-" * 40

# =============================================================================
# Test Functions
# =============================================================================

def test_client_initialization() -> bool:
    """Test Wikidata client initialization and configuration."""
    print(f"\n🔧 Testing Client Initialization")
    print(TestConfig.SUBSEPARATOR)
    
    try:
        client = get_thread_local_client()
        user_agent = dict(client.opener.addheaders).get("User-Agent", "Not found")
        
        print(f"✅ Client initialized successfully")
        print(f"   User Agent: {user_agent}")
        
        return True
    except Exception as e:
        print(f"❌ Client initialization failed: {e}")
        return False


def test_entity_search() -> bool:
    """Test entity search functionality."""
    print(f"\n🔍 Testing Entity Search")
    print(TestConfig.SUBSEPARATOR)
    
    try:
        entities = search_wikidata_relevant_id(TestConfig.QUERY_TEXT, topk=TestConfig.SEARCH_LIMIT)
        
        print(f"✅ Search completed for '{TestConfig.QUERY_TEXT}'")
        print(f"   Found {len(entities)} results:")
        
        for i, entity in enumerate(entities, 1):
            print(f"\n   Result {i}:")
            for key, value in entity.items():
                print(f"     {key}: {value}")
        
        return len(entities) > 0
    except Exception as e:
        print(f"❌ Entity search failed: {e}")
        return False


def test_entity_details_and_triplets() -> bool:
    """Test entity details retrieval and triplet extraction."""
    print(f"\n📊 Testing Entity Details and Triplets")
    print(TestConfig.SUBSEPARATOR)
    
    try:
        # Get entity details
        info = fetch_entity_details(TestConfig.QUERY_ENTITY)
        print(f"✅ Entity details retrieved for {TestConfig.QUERY_ENTITY}")
        print(f"   Entity: {info.get('Title', 'Unknown')} ({TestConfig.QUERY_ENTITY})")
        
        # Display key information
        key_fields = ['Title', 'Description', 'URL']
        for field in key_fields:
            if field in info:
                print(f"   {field}: {info[field]}")
        
        # Get triplets
        triplets, _, _ = fetch_head_entity_triplets(
            TestConfig.QUERY_ENTITY, 
            limit=TestConfig.TRIPLET_LIMIT, 
            mode='ignore'
        )
        
        print(f"\n✅ Retrieved {len(triplets)} triplets:")
        for i, triplet in enumerate(list(triplets)[:5], 1):  # Show first 5
            print(f"   {i}. {triplet}")
        
        if len(triplets) > 5:
            print(f"   ... and {len(triplets) - 5} more triplets")
        
        return len(triplets) > 0
    except Exception as e:
        print(f"❌ Entity details/triplets test failed: {e}")
        return False


def test_property_details_and_triplets() -> bool:
    """Test property details retrieval and triplet extraction."""
    print(f"\n🔗 Testing Property Details and Triplets")
    print(TestConfig.SUBSEPARATOR)
    
    try:
        # Get property details
        info = fetch_relationship_details(TestConfig.QUERY_PROPERTY)
        print(f"✅ Property details retrieved for {TestConfig.QUERY_PROPERTY}")
        print(f"   Property: {info.get('Title', 'Unknown')} ({TestConfig.QUERY_PROPERTY})")
        
        # Display key information
        key_fields = ['Title', 'Description']
        for field in key_fields:
            if field in info:
                print(f"   {field}: {info[field]}")
        
        # Get triplets
        triplets, _ = fetch_relationship_triplet(TestConfig.QUERY_PROPERTY, limit=TestConfig.TRIPLET_LIMIT)
        
        print(f"\n✅ Retrieved {len(triplets)} triplets:")
        for i, triplet in enumerate(list(triplets)[:5], 1):  # Show first 5
            print(f"   {i}. {triplet}")
        
        if len(triplets) > 5:
            print(f"   ... and {len(triplets) - 5} more triplets")
        
        return len(triplets) > 0
    except Exception as e:
        print(f"❌ Property details/triplets test failed: {e}")
        return False


# =============================================================================
# Main Test Execution
# =============================================================================

def run_integrity_tests() -> bool:
    """Run all basic function tests and return overall success."""
    print("🚀 Wikidata Basic Functions Test Suite")
    print(TestConfig.SEPARATOR)
    
    # Define test functions
    tests = [
        ("Client Initialization", test_client_initialization),
        ("Entity Search", test_entity_search),
        ("Entity Details & Triplets", test_entity_details_and_triplets),
        ("Property Details & Triplets", test_property_details_and_triplets),
    ]
    
    # Run tests
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Display results
    print(f"\n{TestConfig.SEPARATOR}")
    print("📊 TEST RESULTS SUMMARY")
    print(TestConfig.SEPARATOR)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if success:
            passed += 1
    
    print(TestConfig.SUBSEPARATOR)
    print(f"Tests Passed: {passed}/{len(results)}")
    
    overall_success = passed == len(results)
    if overall_success:
        print("🎉 All basic function tests passed! Core Wikidata utilities are working correctly.")
    else:
        print("💥 Some basic function tests failed. Please check the errors above.")
    
    return overall_success


if __name__ == '__main__':
    success = run_integrity_tests()
    sys.exit(0 if success else 1)
