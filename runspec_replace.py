#!/usr/bin/env python3
import json
import os
import glob
from pathlib import Path

def process_translation_runspecs():
    """
    Find all runspecs with translation tasks, copy them to translation.json without the tasks attribute.
    """
    # Get the base directory for the runspecs
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    runspecs_dir = base_dir / 'runspecs'
    translation_json_path = runspecs_dir / 'translation.json'
    
    # Initialize dictionary to store runspecs
    translation_runspecs = {}
    
    # Search through all JSON files in the runspecs directory
    for json_file in glob.glob(str(runspecs_dir / '*.json')):
        # Skip translation.json itself
        if os.path.basename(json_file) == 'translation.json':
            continue
            
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Check if this is a collection of runspecs (dictionary of dictionaries)
            if isinstance(data, dict):
                # Process each runspec in the file
                for runspec_id, runspec in data.items():
                    if isinstance(runspec, dict):
                        # Handle case where tasks is a list
                        if isinstance(runspec.get('tasks'), list) and 'Translation' in runspec.get('tasks'):
                            # Copy the runspec without the tasks attribute
                            clean_runspec = {k: v for k, v in runspec.items() if k != 'tasks'}
                            translation_runspecs[runspec_id] = clean_runspec
                        # Handle case where tasks is a string
                        elif runspec.get('tasks') == 'Translation':
                            # Copy the runspec without the tasks attribute
                            clean_runspec = {k: v for k, v in runspec.items() if k != 'tasks'}
                            translation_runspecs[runspec_id] = clean_runspec
            
        except json.JSONDecodeError:
            print(f"Error parsing {json_file}: Invalid JSON format")
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    # Write to translation.json
    with open(translation_json_path, 'w', encoding='utf-8') as f:
        json.dump(translation_runspecs, f, indent=2)
    
    print(f"Updated {translation_json_path} with {len(translation_runspecs)} translation runspecs")

if __name__ == "__main__":
    process_translation_runspecs()
