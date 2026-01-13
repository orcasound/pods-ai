#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Extract training samples from detections.csv with the following constraints:
- At least 30 samples per category (or all available if < 30)
- At least one sample per category-node combination
- Prefer tp_machine_only or fp_machine_only notes
- Spread samples evenly across nodes per category
- Minimize total rows while meeting constraints
- Subtract 2 seconds from timestamps
"""

import csv
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple
from pytz import timezone

PACIFIC_TZ = timezone('US/Pacific')
PREFERRED_NOTES = {'tp_machine_only', 'fp_machine_only'}
MIN_SAMPLES_PER_CATEGORY = 30


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse a PST timestamp string to datetime object."""
    dt = datetime.strptime(timestamp_str, '%Y_%m_%d_%H_%M_%S_PST')
    return PACIFIC_TZ.localize(dt)


def format_timestamp(dt: datetime) -> str:
    """Format a datetime object to PST timestamp string."""
    return dt.strftime('%Y_%m_%d_%H_%M_%S_PST')


def subtract_two_seconds(timestamp_str: str) -> str:
    """Subtract 2 seconds from a PST timestamp string."""
    dt = parse_timestamp(timestamp_str)
    dt_minus_2 = dt - timedelta(seconds=2)
    return format_timestamp(dt_minus_2)


def load_detections(csv_path: Path) -> List[Dict]:
    """Load all detections from CSV file."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def organize_by_category_node(detections: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Organize detections by category and node.
    Returns: {category: {node: [list of detections]}}
    """
    organized = defaultdict(lambda: defaultdict(list))
    for det in detections:
        category = det['Category']
        node = det['NodeName']
        organized[category][node].append(det)
    return organized


def sort_by_preference(detections: List[Dict]) -> List[Dict]:
    """
    Sort detections by preference:
    1. Preferred notes (tp_machine_only, fp_machine_only) first
    2. Descriptions without "faint" or "distant" preferred
    3. Then by timestamp (oldest first)
    """
    def sort_key(det):
        has_preferred_note = det['Notes'] in PREFERRED_NOTES
        description = det.get('Description', '').lower()
        has_unpreferred_description = 'faint' in description or 'distant' in description
        timestamp = det['Timestamp']
        # Return tuple: (not preferred note, has unpreferred description, timestamp)
        # This puts preferred notes first, then non-faint/distant descriptions, then by timestamp
        return (not has_preferred_note, has_unpreferred_description, timestamp)
    
    return sorted(detections, key=sort_key)


def select_training_samples(organized_data: Dict[str, Dict[str, List[Dict]]]) -> List[Dict]:
    """
    Select training samples according to requirements:
    - At least 30 samples per category (or all if < 30)
    - At least one sample per category-node combination
    - Spread evenly across nodes
    - Prefer certain note types
    - Minimize total rows
    """
    selected = []
    
    for category in sorted(organized_data.keys()):
        nodes_data = organized_data[category]
        
        # Sort and prepare detections for each node
        node_detections = {}
        for node in nodes_data.keys():
            node_detections[node] = sort_by_preference(nodes_data[node])
        
        # Calculate target count for this category
        total_available = sum(len(node_detections[node]) for node in node_detections)
        target_count = min(MIN_SAMPLES_PER_CATEGORY, total_available)
        
        # Track how many samples selected per node
        node_counts = defaultdict(int)
        category_samples = []
        
        # Round-robin selection across nodes to ensure even distribution
        # First, ensure at least one sample per node
        for node in sorted(node_detections.keys()):
            if node_detections[node]:
                category_samples.append(node_detections[node][0])
                node_counts[node] = 1
        
        # Continue round-robin to reach target count
        while len(category_samples) < target_count:
            added_this_round = False
            
            # Sort nodes by how many samples they've contributed (fewest first)
            # This ensures even distribution
            nodes_by_count = sorted(node_detections.keys(), 
                                   key=lambda n: node_counts[n])
            
            for node in nodes_by_count:
                if len(category_samples) >= target_count:
                    break
                    
                # Check if this node has more samples to give
                if node_counts[node] < len(node_detections[node]):
                    idx = node_counts[node]
                    category_samples.append(node_detections[node][idx])
                    node_counts[node] += 1
                    added_this_round = True
            
            # If no node could contribute, we've exhausted all samples
            if not added_this_round:
                break
        
        # Add all selected samples for this category
        selected.extend(category_samples)
    
    return selected


def write_training_samples(samples: List[Dict], output_path: Path):
    """Write selected samples to CSV with timestamps adjusted."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        # Use same columns as detections.csv
        fieldnames = ['Category', 'NodeName', 'Timestamp', 'URI', 'Description', 'Notes']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for sample in samples:
            # Create a copy and adjust timestamp
            output_row = sample.copy()
            output_row['Timestamp'] = subtract_two_seconds(sample['Timestamp'])
            writer.writerow(output_row)


def main():
    """Main function to extract training samples."""
    # Paths
    input_path = Path('output_segments/detections.csv')
    output_path = Path('output_segments/training_samples.csv')
    
    print(f"Loading detections from {input_path}...")
    detections = load_detections(input_path)
    print(f"Loaded {len(detections)} detections")
    
    print("\nOrganizing detections by category and node...")
    organized_data = organize_by_category_node(detections)
    
    # Print summary
    for category in sorted(organized_data.keys()):
        total = sum(len(nodes) for nodes in organized_data[category].values())
        print(f"  {category}: {total} detections across {len(organized_data[category])} nodes")
    
    print("\nSelecting training samples...")
    samples = select_training_samples(organized_data)
    
    print(f"\nSelected {len(samples)} training samples")
    
    # Print breakdown by category
    category_counts = defaultdict(int)
    category_node_counts = defaultdict(lambda: defaultdict(int))
    for sample in samples:
        category_counts[sample['Category']] += 1
        category_node_counts[sample['Category']][sample['NodeName']] += 1
    
    for category in sorted(category_counts.keys()):
        print(f"  {category}: {category_counts[category]} samples")
        for node in sorted(category_node_counts[category].keys()):
            print(f"    {node}: {category_node_counts[category][node]}")
    
    print(f"\nWriting training samples to {output_path}...")
    write_training_samples(samples, output_path)
    print("Done!")


if __name__ == "__main__":
    main()
