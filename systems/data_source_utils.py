# -*- coding: utf-8 -*-
"""
Utility functions for expanding data source patterns to actual file paths.

This module provides shared logic for matching file patterns (with wildcards,
fuzzy matching, etc.) against files in a dataset directory.
"""

import fnmatch
import glob
import os
from typing import Dict, List, Optional


def expand_data_sources(
    data_sources: List[str],
    dataset_directory: str,
    all_files: List[str],
    verbose: bool = False
) -> List[str]:
    """
    Expand wildcard patterns in data_sources to actual file paths.

    Handles:
    - Wildcards like "State MSA Identity Theft Data/*" or "file-*.csv"
    - Empty string or "./" meaning all files
    - Directory paths ending with "/"
    - Fuzzy names like "Constitution Beach" matching "constitution_beach_datasheet.csv"
    - Case-insensitive matching

    Args:
        data_sources: List of file patterns (may contain wildcards)
        dataset_directory: Base directory containing the dataset files
        all_files: List of all file paths relative to dataset_directory
        verbose: Whether to print warnings for unmatched patterns

    Returns:
        List of actual file paths (relative to current working directory)
    """
    if not dataset_directory:
        return []

    # Handle empty data_sources or special "all files" patterns
    if not data_sources:
        return []

    expanded_paths = []

    for pattern in data_sources:
        # Handle empty string or "./" as "all files"
        if pattern == "" or pattern == "./" or pattern == ".":
            expanded_paths.extend(all_files)
            continue

        # Handle directory paths ending with "/" - get all files in that directory
        if pattern.endswith('/'):
            dir_pattern = pattern.rstrip('/')
            for f in all_files:
                if f.startswith(dir_pattern + '/') or dir_pattern in f:
                    expanded_paths.append(f)
            continue

        # Check if pattern contains wildcards
        if '*' in pattern or '?' in pattern:
            matched = _match_wildcard_pattern(pattern, all_files, dataset_directory)
            if matched:
                expanded_paths.extend(matched)
            elif verbose:
                print(f"WARNING: No files matched pattern '{pattern}'")
        else:
            # No wildcards - treat as exact path or fuzzy match
            matched = _match_exact_or_fuzzy(pattern, all_files, dataset_directory)
            if matched:
                expanded_paths.extend(matched)
            elif verbose:
                print(f"WARNING: File not found '{pattern}'")

    # Convert to paths relative to cwd for agent use
    result = []
    for p in expanded_paths:
        full_path = os.path.join(dataset_directory, p)
        result.append(os.path.relpath(full_path))

    return list(set(result))  # Remove duplicates


def _match_wildcard_pattern(
    pattern: str,
    all_files: List[str],
    dataset_directory: str
) -> List[str]:
    """
    Match a wildcard pattern against all files.

    Args:
        pattern: Pattern containing * or ? wildcards
        all_files: List of all file paths relative to dataset_directory
        dataset_directory: Base directory for glob fallback

    Returns:
        List of matched file paths (relative to dataset_directory)
    """
    matched = []
    pattern_lower = pattern.lower()
    pattern_dir_lower = os.path.dirname(pattern).lower()
    pattern_base_lower = os.path.basename(pattern).lower()

    for f in all_files:
        f_lower = f.lower()
        # Match against full relative path (case-insensitive)
        if fnmatch.fnmatch(f_lower, pattern_lower) or fnmatch.fnmatch(f_lower, f"**/{pattern_lower}"):
            matched.append(f)
        # Also try matching basename against pattern's basename (case-insensitive)
        elif fnmatch.fnmatch(os.path.basename(f_lower), pattern_base_lower):
            # Check if parent directory matches too (case-insensitive)
            if not pattern_dir_lower or pattern_dir_lower in f_lower:
                matched.append(f)

    if matched:
        return matched

    # Fallback: try glob with recursive search
    glob_pattern = os.path.join(dataset_directory, "**", pattern)
    glob_matches = glob.glob(glob_pattern, recursive=True)
    return [os.path.relpath(match, dataset_directory) for match in glob_matches]


def _match_exact_or_fuzzy(
    pattern: str,
    all_files: List[str],
    dataset_directory: str
) -> List[str]:
    """
    Match a pattern without wildcards using exact or fuzzy matching.

    Args:
        pattern: File path or name pattern (no wildcards)
        all_files: List of all file paths relative to dataset_directory
        dataset_directory: Base directory for path resolution

    Returns:
        List of matched file paths (relative to dataset_directory)
    """
    matched = []
    exact_path = os.path.join(dataset_directory, pattern)

    if os.path.exists(exact_path):
        # Check if it's a directory
        if os.path.isdir(exact_path):
            for f in all_files:
                if f.startswith(pattern + '/') or f.startswith(pattern + os.sep):
                    matched.append(f)
        else:
            matched.append(pattern)
        return matched

    # Search for file anywhere in dataset with fuzzy matching
    # Normalize pattern for fuzzy matching (e.g., "Constitution Beach" -> "constitution_beach")
    pattern_normalized = pattern.lower().replace(' ', '_').replace('-', '_')

    for f in all_files:
        # Exact suffix match
        if f.endswith(pattern) or os.path.basename(f) == pattern:
            matched.append(f)
        # Fuzzy match: check if normalized pattern is in the file path
        elif pattern_normalized in f.lower().replace(' ', '_').replace('-', '_'):
            matched.append(f)

    return matched


def check_data_source_exists(
    data_source: str,
    dataset_directory: str,
    all_files: List[str]
) -> bool:
    """
    Check if a data source pattern matches any files.

    Args:
        data_source: A single file pattern
        dataset_directory: Base directory containing the dataset files
        all_files: List of all file paths relative to dataset_directory

    Returns:
        True if the pattern matches at least one file, False otherwise
    """
    if not data_source or data_source in ("", "./", "."):
        return True  # "all files" patterns are always valid

    if data_source.endswith('/'):
        dir_pattern = data_source.rstrip('/')
        for f in all_files:
            if f.startswith(dir_pattern + '/') or dir_pattern in f:
                return True
        return False

    if '*' in data_source or '?' in data_source:
        matched = _match_wildcard_pattern(data_source, all_files, dataset_directory)
        return len(matched) > 0
    else:
        matched = _match_exact_or_fuzzy(data_source, all_files, dataset_directory)
        return len(matched) > 0


def get_dataset_files(dataset_directory: str) -> Dict[str, None]:
    """
    Collect all files in a dataset directory.

    Args:
        dataset_directory: Path to the dataset directory

    Returns:
        Dictionary mapping relative file paths to None (placeholder for content)
    """
    dataset = {}
    for dirpath, _, filenames in os.walk(dataset_directory):
        for fname in filenames:
            rel_path = os.path.relpath(
                os.path.join(dirpath, fname), dataset_directory
            )
            dataset[rel_path] = None
    return dataset
