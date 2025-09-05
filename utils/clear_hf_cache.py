#!/usr/bin/env python3
"""Utility script to clear HuggingFace caches and temporary files.

This module provides functions to clear datasets cache, models cache,
and temporary HuggingFace files to free up disk space.
"""

import glob
import logging
import os
import shutil
import subprocess
import tempfile

from datasets import config as datasets_config
from huggingface_hub import constants as hf_hub_constants

logger = logging.getLogger(__name__)


def clear_datasets_cache() -> None:
    """Clear the HuggingFace datasets cache directory.
    
    Removes all files in the datasets cache directory and logs
    the total size of files that were removed.
    """
    cache_dir = datasets_config.HF_DATASETS_CACHE
    logger.info("Datasets cache directory: %s", cache_dir)

    if os.path.exists(cache_dir):
        logger.info("Clearing datasets cache at: %s", cache_dir)
        try:
            # Get size before deletion
            total_size = 0
            file_count = 0
            for dirpath, _, filenames in os.walk(cache_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total_size += os.path.getsize(fp)
                        file_count += 1

            logger.info("Found %d files totaling %.2f GB", file_count, total_size / (1024 * 1024 * 1024))

            # Delete the cache
            shutil.rmtree(cache_dir, ignore_errors=True)
            logger.info("Datasets cache cleared successfully.")
        except (OSError, FileNotFoundError) as exc:
            logger.error("Error clearing datasets cache: %s", exc)
    else:
        logger.info("No datasets cache found.")


def clear_models_cache() -> None:
    """Clear the HuggingFace models cache directory.
    
    Removes all files in the models cache directory and logs
    the total size of files that were removed.
    """
    try:
        models_cache = hf_hub_constants.HF_HUB_CACHE

        logger.info("Models cache directory: %s", models_cache)
        if os.path.exists(models_cache):
            logger.info("Clearing models cache at: %s", models_cache)
            try:
                # Get size before deletion
                total_size = 0
                file_count = 0
                for dirpath, _, filenames in os.walk(models_cache):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            total_size += os.path.getsize(fp)
                            file_count += 1

                logger.info("Found %d files totaling %.2f GB", file_count, total_size / (1024 * 1024 * 1024))

                shutil.rmtree(models_cache, ignore_errors=True)
                logger.info("Models cache cleared successfully.")
            except (OSError, FileNotFoundError) as exc:
                logger.error("Error clearing models cache: %s", exc)
        else:
            logger.info("No models cache found.")
    except (OSError, AttributeError) as exc:
        logger.error("Error clearing models cache: %s", exc)


def clear_temp_files() -> None:
    """Clear temporary HuggingFace files from the system temp directory.
    
    Removes all temporary files and directories with the 'tmphf_' prefix
    from the system's temporary directory.
    """
    temp_dir = tempfile.gettempdir()
    logger.info("Temp directory: %s", temp_dir)

    hf_temp_pattern = os.path.join(temp_dir, "tmphf_*")
    hf_temp_files = glob.glob(hf_temp_pattern)

    if hf_temp_files:
        logger.info("Found %d temporary HF files", len(hf_temp_files))
        for file in hf_temp_files:
            try:
                if os.path.isdir(file):
                    shutil.rmtree(file, ignore_errors=True)
                else:
                    os.remove(file)
                logger.info("Removed: %s", file)
            except (OSError, FileNotFoundError) as exc:
                logger.error("Error removing temp file %s: %s", file, exc)
    else:
        logger.info("No HF temporary files found.")


def print_disk_usage() -> None:
    """Print current disk usage information."""
    try:
        logger.info("Disk usage before cleaning:")
        result = subprocess.run(['df', '-h'], stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, check=True, timeout=30)
        for line in result.stdout.decode('utf-8').splitlines():
            logger.info(line)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.error("Error getting disk usage: %s", exc)


def main() -> None:
    """Main function to run the complete cache cleanup process.
    
    Clears datasets cache, models cache, and temporary files,
    then prints disk usage before and after cleanup.
    """
    logger.info("Starting cache cleanup process")

    print_disk_usage()

    clear_datasets_cache()
    clear_models_cache()
    clear_temp_files()

    logger.info("Cache cleanup completed")
    print_disk_usage()


if __name__ == "__main__":
    main()
