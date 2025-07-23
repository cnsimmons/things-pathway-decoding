#!/usr/bin/env python3
"""
Load preprocessed THINGs data and extract ROI data for decoding.
"""

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from nilearn import datasets, masking
import pickle

def get_roi_masks():
    """Create simple ROI masks using Harvard-Oxford atlas."""
    ho_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    
    # Simple ROI definitions - we can refine these later
    rois = {
        'dorsal_left': [22, 23],    # Superior/Inferior Parietal Lobule left
        'dorsal_right': [41, 42],   # Superior/Inferior Parietal Lobule right  
        'ventral_left': [36, 37],   # Temporal Fusiform, IT left
        'ventral_right': [55, 56],  # Temporal Fusiform, IT right
    }
    
    masks = {}
    for roi_name, indices in rois.items():
        mask_data = np.isin(ho_atlas.maps.get_fdata(), indices)
        masks[roi_name] = ho_atlas.maps.__class__(mask_data.astype(int), 
                                                 ho_atlas.maps.affine)
    
    return masks

def load_things_data(subject_id, session_id, data_dir="./data"):
    """Load THINGs preprocessed beta maps and labels."""
    
    # Load beta maps (already preprocessed!)
    beta_dir = Path(data_dir) / "ds004192" / "derivatives" / "glmsingle" / f"sub-{subject_id}" / f"ses-{session_id}"
    beta_file = beta_dir / "betas_assumehrf.nii.gz"  # or similar filename
    
    if not beta_file.exists():
        # Find actual beta file
        beta_files = list(beta_dir.glob("*beta*.nii.gz"))
        if beta_files:
            beta_file = beta_files[0]
        else:
            raise FileNotFoundError(f"No beta files in {beta_dir}")
    
    beta_img = nib.load(beta_file)
    
    # Load stimulus labels
    events_file = Path(data_dir) / "ds004192" / f"sub-{subject_id}" / f"ses-{session_id}" / "func" / f"sub-{subject_id}_ses-{session_id}_task-things_events.tsv"
    events_df = pd.read_csv(events_file, sep='\t')
    
    return beta_img, events_df['stimulus_name'].values

def process_subject(subject_id, session_id):
    """Simple processing - just extract ROI data from preprocessed betas."""
    
    masks = get_roi_masks()
    beta_img, labels = load_things_data(subject_id, session_id)
    
    # Extract data for each ROI
    roi_data = {}
    for roi_name, mask in masks.items():
        data = masking.apply_mask(beta_img, mask)
        roi_data[roi_name] = {'data': data, 'labels': labels}
    
    # Save
    output_file = Path("results") / f"sub-{subject_id}_ses-{session_id}_roi_data.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(roi_data, f)
    
    print(f"Saved data for sub-{subject_id}/ses-{session_id}")

if __name__ == "__main__":
    subject_id, session_id = sys.argv[1], sys.argv[2]
    process_subject(subject_id, session_id)