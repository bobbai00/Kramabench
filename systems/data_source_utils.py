# -*- coding: utf-8 -*-
"""
Utility functions for expanding data source patterns to actual file paths.

This module provides shared logic for matching file patterns against files
in a dataset directory. Wildcards are passed through as-is for the agent
to interpret.
"""

import os
from typing import Dict, List, Optional, Tuple


def expand_data_sources(
    data_sources: List[str],
    dataset_directory: str,
    all_files: List[str],
    verbose: bool = False
) -> List[str]:
    """
    Process data_sources to actual file paths or pass through wildcards.

    Handles:
    - Exact file names - search and locate in dataset, return full relative path
    - Wildcards like "folder/*" - search for folder, return full path with wildcard
    - Empty string or "./" meaning all files
    - Fuzzy names like "Constitution Beach" matching "constitution_beach_datasheet.csv"

    Args:
        data_sources: List of file patterns (may contain wildcards)
        dataset_directory: Base directory containing the dataset files
        all_files: List of all file paths relative to dataset_directory
        verbose: Whether to print warnings for unmatched patterns

    Returns:
        List of file paths (relative to current working directory)
    """
    if not dataset_directory:
        return []

    if not data_sources:
        return []

    # All paths in this list are relative to dataset_directory
    paths_relative_to_dataset = []

    for pattern in data_sources:
        # Handle empty string or "./" as "all files"
        if pattern == "" or pattern == "./" or pattern == ".":
            paths_relative_to_dataset.extend(all_files)
            continue

        # Handle wildcards - search for the directory part, keep wildcard
        if '*' in pattern or '?' in pattern:
            resolved = _resolve_wildcard_path(pattern, all_files, dataset_directory, verbose)
            paths_relative_to_dataset.append(resolved)
            continue

        # Handle directory paths ending with "/" - convert to wildcard
        if pattern.endswith('/'):
            resolved = _resolve_wildcard_path(pattern + "*", all_files, dataset_directory, verbose)
            paths_relative_to_dataset.append(resolved)
            continue

        # No wildcards - search for exact file or fuzzy match
        matched = _search_and_match(pattern, all_files, dataset_directory)
        if matched:
            paths_relative_to_dataset.extend(matched)
        else:
            # Fallback: just use the pattern as-is
            if verbose:
                print(f"WARNING: File not found '{pattern}', using as-is")
            paths_relative_to_dataset.append(pattern)

    # Convert all paths to be relative to cwd by prepending dataset_directory
    result = []
    for p in paths_relative_to_dataset:
        full_path = os.path.join(dataset_directory, p)
        result.append(os.path.relpath(full_path))

    return list(set(result))  # Remove duplicates


def _resolve_wildcard_path(
    pattern: str,
    all_files: List[str],
    dataset_directory: str,
    verbose: bool = False
) -> str:
    """
    Resolve a wildcard pattern by searching for the directory part.

    Args:
        pattern: Pattern with wildcards (e.g., "State MSA Identity Theft Data/*")
        all_files: List of all file paths relative to dataset_directory
        dataset_directory: Base directory for path resolution

    Returns:
        Resolved pattern with valid directory path, or original pattern if not found
    """
    # Split into directory part and wildcard part
    # e.g., "folder/subfolder/*.csv" -> dir_part="folder/subfolder", wildcard_part="*.csv"
    parts = pattern.rsplit('/', 1)
    if len(parts) == 1:
        # No directory, just a wildcard like "*.csv"
        return pattern

    dir_part, wildcard_part = parts

    # Check if the directory exists directly
    full_dir = os.path.join(dataset_directory, dir_part)
    if os.path.isdir(full_dir):
        return pattern  # Already valid

    # Search for the directory in all_files
    dir_name = os.path.basename(dir_part)
    dir_name_lower = dir_name.lower().replace(' ', '_').replace('-', '_')

    # Look for directories that contain files
    found_dirs = set()
    for f in all_files:
        f_dir = os.path.dirname(f)
        if f_dir:
            # Check each component of the path
            dir_parts = f_dir.split('/')
            for i, part in enumerate(dir_parts):
                part_normalized = part.lower().replace(' ', '_').replace('-', '_')
                if part == dir_name or part_normalized == dir_name_lower:
                    # Found matching directory
                    found_path = '/'.join(dir_parts[:i+1])
                    found_dirs.add(found_path)

    if found_dirs:
        # Use the first match
        found_dir = sorted(found_dirs)[0]
        if verbose:
            print(f"Resolved '{dir_part}' to '{found_dir}'")
        return f"{found_dir}/{wildcard_part}"

    # Not found - return original pattern as fallback
    if verbose:
        print(f"WARNING: Directory '{dir_part}' not found, using pattern as-is")
    return pattern


def _search_and_match(
    pattern: str,
    all_files: List[str],
    dataset_directory: str
) -> List[str]:
    """
    Search for a file or directory by name and return full relative paths.

    Args:
        pattern: File path or name pattern (no wildcards)
        all_files: List of all file paths relative to dataset_directory
        dataset_directory: Base directory for path resolution

    Returns:
        List of matched file paths (relative to dataset_directory)
    """
    matched = []

    # First, check if the exact path exists
    exact_path = os.path.join(dataset_directory, pattern)
    if os.path.exists(exact_path):
        if os.path.isdir(exact_path):
            # Return all files in this directory
            for f in all_files:
                if f.startswith(pattern + '/') or f.startswith(pattern + os.sep):
                    matched.append(f)
        else:
            matched.append(pattern)
        return matched

    # Search for file by basename in all_files
    pattern_basename = os.path.basename(pattern)
    pattern_normalized = pattern.lower().replace(' ', '_').replace('-', '_')
    basename_normalized = pattern_basename.lower().replace(' ', '_').replace('-', '_')

    for f in all_files:
        f_basename = os.path.basename(f)
        f_basename_normalized = f_basename.lower().replace(' ', '_').replace('-', '_')

        # Exact basename match
        if f_basename == pattern_basename:
            matched.append(f)
        # Exact path suffix match
        elif f.endswith(pattern) or f.endswith('/' + pattern):
            matched.append(f)
        # Fuzzy basename match
        elif basename_normalized == f_basename_normalized:
            matched.append(f)
        # Fuzzy substring match in path
        elif pattern_normalized in f.lower().replace(' ', '_').replace('-', '_'):
            matched.append(f)

    return matched


def check_data_source_exists(
    data_source: str,
    dataset_directory: str,
    all_files: List[str]
) -> bool:
    """
    Check if a data source pattern is valid.

    Args:
        data_source: A single file pattern
        dataset_directory: Base directory containing the dataset files
        all_files: List of all file paths relative to dataset_directory

    Returns:
        True if the pattern is valid (wildcards are always valid), False otherwise
    """
    if not data_source or data_source in ("", "./", "."):
        return True  # "all files" patterns are always valid

    # Wildcards and directory patterns are always valid (passed through to agent)
    if '*' in data_source or '?' in data_source or data_source.endswith('/'):
        return True

    # For exact patterns, check if file exists
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
