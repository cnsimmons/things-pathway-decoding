#!/usr/bin/env python3
"""
Download THINGs fMRI data using datalad.
Downloads only the preprocessed single-trial beta maps we need.
"""

import os
import sys
import yaml
import datalad.api as dl
from pathlib import Path

def load_config():
    """Load analysis configuration."""
    config_path = Path(__file__).parent.parent / "configs" / "analysis_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_subject_data(subject_id, session_id, data_dir="./data"):
    """Download single subject/session data."""
    print(f"Downloading subject {subject_id}, session {session_id}...")
    
    config = load_config()
    
    # Create data directory
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    # Install dataset if not already present
    dataset_path = data_dir / "ds004192"
    if not dataset_path.exists():
        print("Installing THINGs dataset...")
        dl.install(str(dataset_path), source=config['dataset']['source'])
    
    # Download specific subject/session beta maps
    beta_path = f"derivatives/glmsingle/sub-{subject_id}/ses-{session_id}/"
    print(f"Getting beta maps: {beta_path}")
    dl.get(str(dataset_path / beta_path))
    
    # Also get stimulus events file
    events_path = f"sub-{subject_id}/ses-{session_id}/func/"
    print(f"Getting events file: {events_path}")
    dl.get(str(dataset_path / events_path))
    
    print(f"Download complete for sub-{subject_id}/ses-{session_id}")
    return dataset_path

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python download_data.py <subject_id> <session_id>")
        print("Example: python download_data.py 01 01")
        sys.exit(1)
    
    subject_id = sys.argv[1]
    session_id = sys.argv[2]
    
    download_subject_data(subject_id, session_id)
