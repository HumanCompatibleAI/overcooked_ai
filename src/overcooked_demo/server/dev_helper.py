"""
Helper module to add the parent src directory to the Python path.
This allows importing human_aware_rl and other modules from the main project.

Usage:
    # At the top of your script (e.g., app.py)
    import os
    import sys
    
    # Add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    
    # Now you can import from human_aware_rl
    from human_aware_rl.rllib.rllib import load_agent
"""

import os
import sys
import pathlib

def add_src_to_path():
    """Add the parent src directory to the Python path."""
    current_dir = pathlib.Path(__file__).parent.absolute()
    src_dir = current_dir.parent.parent  # ../.. relative to this file
    sys.path.insert(0, str(src_dir))
    print(f"Added {src_dir} to Python path")

# Automatically add to path when imported
add_src_to_path() 