#!/usr/bin/env python3
"""
Blacklist Analysis and Cross-Reference Utility

This utility script provides analysis tools for examining blacklist effectiveness
and cross-referencing blacklisted samples with various data patch categories.
It helps researchers understand the impact of data quality measures and validate
blacklist generation results.

Key responsibilities:
- Latest blacklist file discovery and loading
- Cross-reference analysis between blacklists and patch files
- Sample overlap detection across different quality categories
- Data quality impact assessment and reporting

Called by:
- Manual execution for blacklist analysis
- Data quality assessment workflows  
- Research analysis of training data modifications
- Debugging blacklist generation effectiveness

Use cases:
- Validate blacklist generation results
- Analyze overlap between different quality categories
- Understand data modification impact on specific samples
- Research data quality effects on model performance

Data patch categories analyzed:
- delete/: Samples marked for removal (blacklisted)
- do_not_blacklist/: Samples protected from removal
- override_text_perfect/: Manually corrected samples
- Multiple patch files within each category

Output analysis:
- Matching sample identification
- Cross-category overlap statistics
- Sample-level detail for manual review
- Quality category distribution analysis

This script provides essential insights into data quality management
effectiveness and helps optimize the patch system for better training
data quality.
"""

import sys

import pandas as pd
import glob
import os

def find_latest_blacklist_file(directory):
    """
    Discovers the most recent blacklist file based on filesystem timestamps.
    
    Blacklist files are generated with timestamp-based naming patterns,
    enabling automatic discovery of the latest analysis results without
    requiring manual file specification.
    
    Called by:
    - Main analysis workflow for automatic blacklist file discovery
    - Batch analysis scripts processing multiple blacklist generations
    
    File naming pattern:
    - Expected format: 'blacklist-*.csv'
    - Typically includes timestamps: 'blacklist-d3sp-20250121-111828.csv'
    - Uses filesystem modification time for ordering
    
    Search strategy:
    - Scans directory for files matching 'blacklist-*.csv' pattern
    - Orders by modification time (most recent first)
    - Returns path to latest file or None if no matches
    
    Error handling:
    - Empty directory: Returns None gracefully
    - No matching files: Returns None without error
    - Access errors: Propagates filesystem exceptions
    
    Args:
        directory (str): Directory path to search for blacklist files
        
    Returns:
        str | None: Path to latest blacklist file, or None if none found
        
    Example:
        latest = find_latest_blacklist_file('output')
        if latest:
            print(f"Analyzing blacklist: {latest}")
        else:
            print("No blacklist files found")
    """
    list_of_files = glob.glob(os.path.join(directory, 'blacklist-*.csv'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file

def print_matching_lines(file1, file2):
    """
    Performs cross-reference analysis between two CSV files based on sample IDs.
    
    This function identifies samples that appear in both files, enabling
    analysis of overlaps between different data quality categories (blacklists,
    overrides, protections, etc.). Essential for understanding data patch
    interactions and effectiveness.
    
    Called by:
    - Main analysis loop for each patch file comparison
    - Data quality assessment workflows
    - Blacklist validation and effectiveness analysis
    
    Analysis methodology:
    1. Load both CSV files with ID columns as strings (handles mixed types)
    2. Extract valid (non-empty) IDs from second file for matching
    3. Perform inner join to find samples present in both files
    4. Display matching samples with complete information from first file
    
    ID handling:
    - Treats IDs as strings to handle mixed numeric/string formats
    - Filters out NaN/empty IDs to prevent false matches
    - Maintains original ID formats in output display
    
    Output format:
    - Header: Identifies source files being compared
    - Table: Complete sample information for matching entries
    - Empty result: Clear "No matching IDs found" message
    
    Use cases:
    - Blacklist effectiveness: Which blacklisted samples are also protected?
    - Override impact: Which corrected samples were later blacklisted?
    - Quality analysis: Sample overlap across different quality categories
    - Validation: Verify expected relationships between patch categories
    
    Error handling:
    - Missing files: Clear error message with file paths
    - Empty CSV files: Graceful handling with informative message
    - Corrupted files: Exception reporting with file context
    - Missing ID columns: Implicit handling via pandas operations
    
    Args:
        file1 (str): Path to primary CSV file (source of displayed data)
        file2 (str): Path to reference CSV file (source of matching IDs)
        
    Output:
        Prints formatted analysis results to stdout:
        - Comparison header
        - Matching sample details (if any)
        - Summary message for empty results
        
    Example output:
        Lines from blacklist-latest.csv with matching IDs in protected.csv:
        id    audio_path                    text_perfect
        1001  /data/audio/sample1.wav      "corrected text"
        1005  /data/audio/sample5.wav      "high quality sample"
        
        No matching IDs found.
    """
    try:
        df1 = pd.read_csv(file1, dtype={'id': str})  # Read first file, treating 'id' as string
        df2 = pd.read_csv(file2, usecols=['id'], dtype={'id': str})  # Read only 'id' from the second file

        # Remove empty IDs from df2 for matching
        valid_ids_df2 = df2['id'].dropna()

        # Merge based on 'id' to get matching rows
        merged_df = pd.merge(df1, valid_ids_df2, on='id', how='inner')

        if not merged_df.empty:
            print(f"Lines from {file1} with matching IDs in {file2}:")
            print(merged_df.to_string(index=False))  # Print the matching lines without index
        else:
            print("No matching IDs found.")

    except FileNotFoundError:
        print(f"Error: One or both of the files were not found.")
    except pd.errors.EmptyDataError:
        print("Error: One or both of the CSV files are empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # === BLACKLIST ANALYSIS WORKFLOW ===
    # This section demonstrates comprehensive blacklist analysis by cross-referencing
    # the latest generated blacklist with all relevant data patch categories.

    # Discover the most recent blacklist file automatically
    latest_blacklist = find_latest_blacklist_file('output')

    if latest_blacklist:
        print(f"Analyzing blacklist file: {latest_blacklist}")
        print("=" * 60)
    else:
        print("No blacklist files found in output directory.")
        sys.exit(1)

    # Define data patch files for comprehensive analysis
    # These represent different data quality categories and manual interventions
    patch_files = [
        # Deletion category: Samples marked for removal
        'data_patches/data3/delete/data3_prepared - remove - translated.csv',

        # Protection category: High-quality samples protected from blacklisting
        'data_patches/data3/do_not_blacklist/blacklist - Keep ground-truth.csv',

        # Override category: Manually corrected transcriptions
        'data_patches/data3/override_text_perfect/blacklist - Keep Edited.csv',
        'data_patches/data3/override_text_perfect/data3_prepared - edited.csv',
    ]

    print("\n=== CROSS-REFERENCE ANALYSIS RESULTS ===\n")

    # Perform cross-reference analysis for each patch category
    for patch_file in patch_files:
        print(f"\n--- Comparing with: {patch_file} ---")

        # Check if patch file exists before analysis
        if os.path.exists(patch_file):
            print_matching_lines(latest_blacklist, patch_file)
        else:
            print(f"Warning: Patch file not found: {patch_file}")

        print("-" * 50)

    print("\n=== ANALYSIS COMPLETE ===\n")
    print("Summary:")
    print("- Cross-referenced latest blacklist with all patch categories")
    print("- Identified overlaps between blacklisted and protected/corrected samples")
    print("- Results show effectiveness of data quality management system")
    print("\nUse these results to:")
    print("- Validate blacklist generation accuracy")
    print("- Identify potential conflicts in data quality decisions")
    print("- Optimize protection and override strategies")
    print("- Assess impact of manual corrections on training data")
