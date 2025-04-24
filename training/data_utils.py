"""
Utilities for handling data loading and splitting.
"""
import os
import random
from pathlib import Path
from typing import List, Tuple
import logging

logger = logging.getLogger("DataUtils")

def get_train_test_filepaths(
    data_dir: str,
    test_split_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Scans a directory for .wav files and splits them into training and testing sets.

    Args:
        data_dir: The root directory containing the .wav files (can be nested).
        test_split_ratio: The proportion of files to allocate to the test set (default: 0.1).
        random_seed: Seed for the random number generator for reproducible splits (default: 42).

    Returns:
        A tuple containing two lists: (train_filepaths, test_filepaths).
        Returns ([], []) if no .wav files are found or data_dir is invalid.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        logger.error(f"Data directory not found or is not a directory: {data_dir}")
        return [], []

    all_filepaths = []
    logger.info(f"Scanning {data_dir} for .wav files...")
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                # Store relative paths from the data_dir for better portability
                full_path = Path(root) / file
                relative_path = str(full_path.relative_to(data_path))
                all_filepaths.append(relative_path)

    if not all_filepaths:
        logger.warning(f"No .wav files found in {data_dir}.")
        return [], []

    logger.info(f"Found {len(all_filepaths)} total .wav files.")

    # Sort before shuffling for consistency across runs if file system order changes
    all_filepaths.sort()

    # Shuffle with a fixed seed for reproducibility
    random.seed(random_seed)
    random.shuffle(all_filepaths)

    # Calculate split index
    split_index = int(len(all_filepaths) * (1 - test_split_ratio))

    train_filepaths = all_filepaths[:split_index]
    test_filepaths = all_filepaths[split_index:]

    logger.info(f"Splitting data: {len(train_filepaths)} train files, {len(test_filepaths)} test files.")

    # Return lists of relative paths
    return train_filepaths, test_filepaths 