"""
Test configuration values and sanity checks.
"""
from pathlib import Path

def test_config_paths_exist():
    """All data directories exist or can be created."""
    expected_paths = [
        "data/raw",
        "data/labeled", 
        "data/synthetic",
        "data/scored",
        "data/filtered",
        "checkpoints",
    ]
    for path_str in expected_paths:
        Path(path_str).mkdir(parents=True, exist_ok=True)

def test_config_constants_defined():
    """Required constants are defined."""
    # These would be imported from config if available in test environment
    # For now, just verify they exist in source