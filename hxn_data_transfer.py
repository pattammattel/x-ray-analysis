#print(f"Loading {__file__!r} ...")
import httpx
import subprocess
import os
import json 
import numpy as np
import matplotlib.pyplot as plt
from hxntools.CompositeBroker import db
import h5py
import os
import glob
import re
from datetime import datetime

def get_proposal_info(proposal_id):
    """
    Fetch detailed proposal info from the NSLS-II API using httpx.

    Parameters
    ----------
    proposal_id : str or int
        NSLS-II proposal number (e.g., 300001)

    Returns
    -------
    dict
        Example:
        {
            "proposal_id": "300001",
            "title": "Study of Sample Dynamics Using X-ray Techniques",
            "cycle": "2025-3",
            "pi_lastname": "Doe",
            "users": [
                {"name": "Jane Doe", "username": "jdoe"},
                {"name": "John Smith", "username": "jsmith"},
                {"name": "Alex Brown", "username": "abrown"}
            ]
        }
    """
    base_url = "https://api.nsls2.bnl.gov/v1"
    proposal_url = f"{base_url}/proposal/{proposal_id}"

    try:
        with httpx.Client(timeout=10.0) as client:
            # --- Fetch proposal info ---
            r = client.get(proposal_url)
            r.raise_for_status()
            proposal_data = r.json().get("proposal", {})

            users = proposal_data.get("users", [])
            cycles = proposal_data.get("cycles", [])
            title = proposal_data.get("title", "Unknown Title")

            # --- Extract PI and user info ---
            pi_lastname = None
            user_list = []
            for user in users:
                name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip()
                username = user.get("username", "")
                user_list.append({"name": name, "username": username})
                if user.get("is_pi"):
                    pi_lastname = user.get("last_name")

            # --- Determine cycle ---
            if cycles:
                current_cycle = cycles[0]
            else:
                try:
                    cycle_r = client.get(f"{base_url}/facility/nsls2/cycles/current")
                    cycle_r.raise_for_status()
                    current_cycle = cycle_r.json().get("cycle", "Unknown")
                except httpx.RequestError:
                    current_cycle = "Unknown"

    except httpx.RequestError as e:
        print(f"Failed to fetch proposal info: {e}")
        return None

    return {
        "proposal_id": str(proposal_id),
        "title": title,
        "cycle": current_cycle,
        "pi_lastname": pi_lastname,
        "users": user_list
    }


def create_local_user_dir(proposal_id):
    """
    Create a user data directory based on proposal ID.

    Fetches proposal info to determine PI last name and cycle, then checks 
    for existing directories in both old and new formats:
        Old: /data/users/<cycle>/<pi_lastname>_<cycle>/  (e.g., 2026Q1)
        New: /nsls2/data/hxn/legacy/users/<cycle>/<pi_lastname>-<cycle>/  (e.g., 2025-3)

    If a directory exists in either format, returns that path.
    If neither exists, creates a new directory using the new format.

    Parameters
    ----------
    proposal_id : str or int
        NSLS-II proposal number (e.g., 312345)

    Returns
    -------
    str or None
        Full path to the created or existing user directory.
        Returns None if proposal info cannot be retrieved.
    """

    # Fetch proposal info
    info = get_proposal_info(proposal_id)
    if not info:
        print(f"Could not fetch proposal info for ID {proposal_id}")
        return None

    pi_lastname = info.get("pi_lastname")
    cycle = info.get("cycle")

    # Validate required info
    if not pi_lastname or not cycle:
        print(f"Missing PI or cycle info in proposal {proposal_id}")
        return None

    # Check for old format directory (e.g., /data/users/2026Q1/Hu_2026Q1/)
    # Convert cycle to old format if possible (e.g., '2025-3' -> '2025Q3')
    old_cycle = cycle.replace('-', 'Q') if '-' in cycle else cycle
    old_dir = f"/data/users/{old_cycle}/{pi_lastname}_{old_cycle}"
    
    if os.path.exists(old_dir):
        print(f"Found existing directory (old format): {old_dir}")
        return old_dir

    # Check for new format directory
    new_dir = f"/nsls2/data/hxn/legacy/users/{cycle}/{pi_lastname}-{cycle}"
    
    if os.path.exists(new_dir):
        print(f"Found existing directory (new format): {new_dir}")
        return new_dir

    # Neither exists - create new format directory
    os.makedirs(new_dir, exist_ok=True)
    print(f"Created user directory (new format): {new_dir}")
    
    return new_dir


def create_user_dir_from_proposal(proposal_id):
    """
    Fetch proposal info and create a local user directory based on the
    PI last name and cycle using create_local_user_dir().

    Also saves the proposal info into a JSON file inside that directory.

    Directory structure:
        /data/users/<cycle>/<pi_lastname>-<cycle>/
            └── proposal_info.json

    Parameters
    ----------
    proposal_id : str or int
        NSLS-II proposal number (e.g., 312345)

    Returns
    -------
    str or None
        Full path to the created or existing user directory.
        Returns None if proposal info cannot be retrieved.
    """

    # --- Create the local directory (also fetches proposal info internally) ---
    dir_path = create_local_user_dir(proposal_id)
    if not dir_path:
        return None

    print(f"Local user directory ready for proposal {proposal_id}: {dir_path}")

    # --- Fetch proposal info again to save JSON ---
    info = get_proposal_info(proposal_id)
    if not info:
        return dir_path  # Directory exists but couldn't save JSON

    # --- Save proposal info JSON ---
    json_path = os.path.join(dir_path, "proposal_info.json")
    try:
        with open(json_path, "w") as f:
            json.dump(info, f, indent=4)
        print(f"Saved proposal info to: {json_path}")
    except Exception as e:
        print(f"Failed to write proposal_info.json: {e}")

    return dir_path



def get_proposal_paths(proposal_id):
    """
    Return the local and proposal data paths for a given NSLS-II proposal ID.

    Directory structure:
        Local:    /data/users/<cycle>/<pi_lastname>-<cycle>/
        Proposal: /nsls2/data/hxn/proposals/<cycle>/pass-<proposal_id>/

    Parameters
    ----------
    proposal_id : str or int
        NSLS-II proposal number (e.g., 312345)

    Returns
    -------
    tuple or None
        (local_path, proposal_path)
        Returns None if proposal info could not be fetched or is incomplete.
    """

    info = get_proposal_info(proposal_id)
    if not info:
        return None

    cycle = info.get("cycle")
    if not cycle:
        return None

    local_path = create_local_user_dir(proposal_id)
    if not local_path:
        return None
        
    proposal_path = f"/nsls2/data/hxn/proposals/{cycle}/pass-{proposal_id}"

    return local_path, proposal_path


def open_gnome_terminal_su_copy(src_dir: str, dst_dir: str):
    """
    Open a GNOME terminal that asks for username using 'su',
    then runs rsync to copy data from src_dir → dst_dir.
    DUO push and password handled in terminal.
    """

    rsync_cmd = f"rsync -avh --progress --stats '{src_dir}' '{dst_dir}'"

    shell_script = (
        "echo '---------------------------------------'; "
        "echo ' HXN Data Copy Utility '; "
        "echo '---------------------------------------'; "
        "echo; "
        "read -p 'Enter your NSLS-II username: ' uname; "
        "echo; "
        "echo \"Logging in as: $uname\"; "
        "echo 'Enter your password (DUO push will be sent after password entry)...'; "
        "echo; "
        f"su - $uname -c \"{rsync_cmd}\"; "
        "EXIT_CODE=$?; "
        "echo; "
        "if [ $EXIT_CODE -eq 0 ]; then "
        "    echo '✓ SUCCESS: Data copy completed successfully!'; "
        "else "
        "    echo '✗ FAILED: Data copy did not complete (exit code: '$EXIT_CODE')'; "
        "fi; "
        "echo 'Press Enter to close this window.'; "
        "read"
    )

    gnome_cmd = [
        "gnome-terminal",
        "--",
        "bash",
        "-c",
        shell_script
    ]

    try:
        subprocess.Popen(gnome_cmd)
        print(f"\n A GNOME terminal has opened.")
        print(f"   From: {src_dir}")
        print(f"   To:   {dst_dir}")
        print("   Log in to complete the copy (DUO will push automatically).\n")
    except Exception as e:
        print(f" Failed to open GNOME terminal:\n{e}")


def copy_data_from_proposal(proposal_id):
    """
    Copy user data for a given proposal to its official proposal directory.

    Uses:
      - get_proposal_paths() → to resolve local and proposal paths
      - open_gnome_terminal_su_copy() → to perform the rsync copy in a new GNOME terminal

    Parameters
    ----------
    proposal_id : str or int
        NSLS-II proposal number (e.g., 312345)
    """

    # --- Get source and destination paths ---
    paths = get_proposal_paths(proposal_id)
    if not paths:
        print(f"Could not determine paths for proposal {proposal_id}")
        return

    local_path, proposal_path = paths

    # --- Check if destination directory exists ---
    if not os.path.exists(proposal_path):
        print(f"ERROR: Destination directory does not exist: {proposal_path}")
        print(f"Please ensure the proposal directory is created before attempting to copy data.")
        return

    print(f" Preparing to copy data for proposal {proposal_id}:")
    print(f"   From: {local_path}")
    print(f"   To:   {proposal_path}\n")

    # --- Launch the interactive rsync copy ---
    open_gnome_terminal_su_copy(local_path, proposal_path)


def create_symlink_in_proposal(proposal_id):
    """
    Create a symbolic link in the proposal directory pointing to the local user directory.

    Directory structure:
        /nsls2/data/hxn/proposals/<cycle>/pass-<proposal_id>/
            └── <pi_lastname>-<cycle> → /nsls2/data/hxn/legacy/users/<cycle>/<pi_lastname>-<cycle>/

    Parameters
    ----------
    proposal_id : str or int
        NSLS-II proposal number (e.g., 312345)

    Returns
    -------
    dict
        Status dictionary with keys:
        - 'success': bool - whether the operation was successful
        - 'message': str - status or error message
        - 'local_path': str or None - source path (symlink target)
        - 'proposal_path': str or None - proposal directory path
        - 'symlink_path': str or None - full path to the created symlink
    """

    # --- Get source and destination paths ---
    paths = get_proposal_paths(proposal_id)
    if not paths:
        return {
            'success': False,
            'message': f"Could not determine paths for proposal {proposal_id}",
            'local_path': None,
            'proposal_path': None,
            'symlink_path': None
        }

    local_path, proposal_path = paths

    # --- Check if proposal directory exists ---
    if not os.path.exists(proposal_path):
        return {
            'success': False,
            'message': f"Proposal directory does not exist:\n{proposal_path}\n\nPlease ensure the proposal directory is created first.",
            'local_path': local_path,
            'proposal_path': proposal_path,
            'symlink_path': None
        }

    # --- Get symlink name from local path basename ---
    symlink_name = os.path.basename(local_path)
    symlink_path = os.path.join(proposal_path, symlink_name)

    # --- Check if symlink already exists ---
    if os.path.exists(symlink_path):
        if os.path.islink(symlink_path):
            existing_target = os.readlink(symlink_path)
            if existing_target == local_path:
                return {
                    'success': True,
                    'message': f"Symlink already exists and points to the correct location.",
                    'local_path': local_path,
                    'proposal_path': proposal_path,
                    'symlink_path': symlink_path
                }
            else:
                return {
                    'success': False,
                    'message': f"Symlink already exists but points to a different location:\n{existing_target}\n\nExpected:\n{local_path}",
                    'local_path': local_path,
                    'proposal_path': proposal_path,
                    'symlink_path': symlink_path
                }
        else:
            return {
                'success': False,
                'message': f"A file or directory already exists at:\n{symlink_path}\n\nCannot create symlink.",
                'local_path': local_path,
                'proposal_path': proposal_path,
                'symlink_path': symlink_path
            }

    # --- Create the symlink ---
    try:
        os.symlink(local_path, symlink_path)
        print(f"✓ Symlink created successfully:")
        print(f"  Link: {symlink_path}")
        print(f"  Target: {local_path}")
        return {
            'success': True,
            'message': f"Symlink created successfully",
            'local_path': local_path,
            'proposal_path': proposal_path,
            'symlink_path': symlink_path
        }
    except OSError as e:
        return {
            'success': False,
            'message': f"Failed to create symlink:\n{str(e)}",
            'local_path': local_path,
            'proposal_path': proposal_path,
            'symlink_path': symlink_path
        }

