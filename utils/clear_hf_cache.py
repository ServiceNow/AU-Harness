#!/usr/bin/env python3

import os
import shutil
import time
from datasets import config as datasets_config
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_datasets_cache():
    cache_dir = datasets_config.HF_DATASETS_CACHE
    logger.info(f"Datasets cache directory: {cache_dir}")
    
    if os.path.exists(cache_dir):
        logger.info(f"Clearing datasets cache at: {cache_dir}")
        try:
            # Get size before deletion
            total_size = 0
            file_count = 0
            for dirpath, dirnames, filenames in os.walk(cache_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total_size += os.path.getsize(fp)
                        file_count += 1
            
            logger.info(f"Found {file_count} files totaling {total_size / (1024*1024*1024):.2f} GB")
            
            # Delete the cache
            shutil.rmtree(cache_dir, ignore_errors=True)
            logger.info("Datasets cache cleared successfully.")
        except Exception as e:
            logger.error(f"Error clearing datasets cache: {e}")
    else:
        logger.info("No datasets cache found.")

def clear_models_cache():
    try:
        from transformers.utils.hub import HUGGINGFACE_HUB_CACHE
        models_cache = HUGGINGFACE_HUB_CACHE
        
        logger.info(f"Models cache directory: {models_cache}")
        if os.path.exists(models_cache):
            logger.info(f"Clearing models cache at: {models_cache}")
            try:
                # Get size before deletion
                total_size = 0
                file_count = 0
                for dirpath, dirnames, filenames in os.walk(models_cache):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            total_size += os.path.getsize(fp)
                            file_count += 1
                
                logger.info(f"Found {file_count} files totaling {total_size / (1024*1024*1024):.2f} GB")
                
                shutil.rmtree(models_cache, ignore_errors=True)
                logger.info("Models cache cleared successfully.")
            except Exception as e:
                logger.error(f"Error clearing models cache: {e}")
        else:
            logger.info("No models cache found.")
    except ImportError:
        logger.warning("Transformers not found, trying alternative methods")
        home = os.path.expanduser("~")
        alt_cache = os.path.join(home, ".cache", "huggingface", "transformers")
        if os.path.exists(alt_cache):
            logger.info(f"Clearing alternative models cache at: {alt_cache}")
            shutil.rmtree(alt_cache, ignore_errors=True)
            logger.info("Alternative models cache cleared successfully.")

def clear_temp_files():
    import tempfile
    temp_dir = tempfile.gettempdir()
    logger.info(f"Temp directory: {temp_dir}")
    
    hf_temp_pattern = os.path.join(temp_dir, "tmphf_*")
    import glob
    hf_temp_files = glob.glob(hf_temp_pattern)
    
    if hf_temp_files:
        logger.info(f"Found {len(hf_temp_files)} temporary HF files")
        for file in hf_temp_files:
            try:
                if os.path.isdir(file):
                    shutil.rmtree(file, ignore_errors=True)
                else:
                    os.remove(file)
                logger.info(f"Removed: {file}")
            except Exception as e:
                logger.error(f"Error removing temp file {file}: {e}")
    else:
        logger.info("No HF temporary files found.")

def print_disk_usage():
    try:
        import subprocess
        logger.info("Disk usage before cleaning:")
        result = subprocess.run(['df', '-h'], stdout=subprocess.PIPE)
        for line in result.stdout.decode('utf-8').splitlines():
            logger.info(line)
    except Exception as e:
        logger.error(f"Error getting disk usage: {e}")

def main():
    logger.info("Starting cache cleanup process")
    
    print_disk_usage()
    
    clear_datasets_cache()
    clear_models_cache()
    clear_temp_files()
    
    logger.info("Cache cleanup completed")
    print_disk_usage()

if __name__ == "__main__":
    main()