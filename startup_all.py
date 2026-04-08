"""
HXN X-ray Analysis Startup Script
==================================

This script loads all analysis functions from the workspace modules.
Automatically loaded when running: pixi run analysis

Author: HXN Beamline
Date: 2026-04-08
"""

print("=" * 70)
print("HXN X-ray Analysis Environment")
print("=" * 70)

# --- Core Libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os
import glob
import re
from datetime import datetime

print("\n✓ Core libraries loaded (numpy, pandas, matplotlib, h5py)")

# --- HXN Tools ---
try:
    from hxntools.CompositeBroker import db
    print("✓ Databroker (db) loaded")
except ImportError:
    db = None
    print("⚠ Databroker not available (offline mode)")

# --- Data Transfer Functions ---
try:
    from hxn_data_transfer import (
        get_proposal_info,
        create_local_user_dir,
        create_user_dir_from_proposal,
        get_proposal_paths,
        copy_data_from_proposal,
        create_symlink_in_proposal
    )
    print("✓ Data transfer functions loaded:")
    print("    • get_proposal_info(proposal_id)")
    print("    • create_local_user_dir(proposal_id)")
    print("    • create_user_dir_from_proposal(proposal_id)")
    print("    • get_proposal_paths(proposal_id)")
    print("    • copy_data_from_proposal(proposal_id)")
    print("    • create_symlink_in_proposal(proposal_id)")
except ImportError as e:
    print(f"⚠ Could not load data transfer functions: {e}")

# --- Scan Metadata Export Functions ---
try:
    from export_scan_details import (
        get_scan_details,
        get_scan_metadata,
        export_scan_details_batch
    )
    print("✓ Scan metadata export functions loaded:")
    print("    • get_scan_details(hdr)")
    print("    • get_scan_metadata(hdr)")
    print("    • export_scan_details_batch(sid_list, wd='.', return_dataframe=False)")
except ImportError as e:
    print(f"⚠ Could not load scan export functions: {e}")

# --- Image Alignment Functions ---
try:
    from alignment import (
        align_stack,
        align_simple,
        align_with_tmat,
        align_stack_iter
    )
    print("✓ Image alignment functions loaded:")
    print("    • align_stack(stack_img, transformation, reference)")
    print("    • align_simple(stack_img, transformation, reference)")
    print("    • align_with_tmat(stack_img, tmat_file, transformation)")
    print("    • align_stack_iter(stack_img, n_iter, transformation, reference)")
except ImportError as e:
    print(f"⚠ Could not load alignment functions: {e}")

# --- 3D Visualization Functions ---
try:
    from view3d_mpl import plot_3d_stack
    print("✓ 3D visualization functions loaded:")
    print("    • plot_3d_stack(data_stack)")
except ImportError as e:
    print(f"⚠ Could not load visualization functions: {e}")

# --- Environment Info ---
print("\n" + "=" * 70)
print("Environment Information")
print("=" * 70)
print(f"Working directory: {os.getcwd()}")
print(f"Python version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {plt.matplotlib.__version__}")

# --- Quick Start Guide ---
print("\n" + "=" * 70)
print("Quick Start Examples")
print("=" * 70)
print("""
# Export scan metadata
export_scan_details_batch([200001, 200002, 200003])

# Get proposal information
info = get_proposal_info(312345)
print(info)

# Create user directory for proposal
path = create_user_dir_from_proposal(312345)

# Create symlink in proposal directory
result = create_symlink_in_proposal(312345)

# Align image stack
aligned = align_simple(stack_img)

# Visualize 3D stack
plot_3d_stack(data_stack)
""")

print("=" * 70)
print("Ready! Start coding...")
print("=" * 70 + "\n")
