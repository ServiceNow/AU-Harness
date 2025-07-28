#!/bin/bash

# Set working directory to the script's location
cd "$(dirname "$0")"
cd ..

# Define target directory and file info
TARGET_DIR="spider"
ZIP_NAME="spider.zip"
FILE_ID="1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"

# Check for gdown
if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Attempting to install..."
    if command -v pip &> /dev/null; then
        pip install gdown
    elif command -v apt &> /dev/null; then
        sudo apt update && sudo apt install -y python3-pip
        pip install gdown
    else
        echo "Error: 'gdown' is not installed and automatic install is not supported on this system."
        exit 1
    fi
fi

# Check for unzip
if ! command -v unzip &> /dev/null; then
    echo "unzip not found. Attempting to install..."
    if command -v apt &> /dev/null; then
        sudo apt update && sudo apt install -y unzip
    else
        echo "Error: 'unzip' is not installed and automatic install is not supported on this system."
        exit 1
    fi
fi

# Check for jq
if ! command -v jq &> /dev/null; then
    echo "jq not found. Attempting to install..."
    if command -v apt &> /dev/null; then
        sudo apt update && sudo apt install -y jq
    else
        echo "Error: 'jq' is not installed and automatic install is not supported on this system."
        exit 1
    fi
fi

# Download and extract if not already downloaded
if ! [ -d "$TARGET_DIR" ]; then
    echo "************************ DOWNLOADING DATASET: Spider ************************"
    mkdir -p "$TARGET_DIR"
    gdown --id "$FILE_ID" -O "$ZIP_NAME"
    unzip "$ZIP_NAME" -d "$TARGET_DIR"
    rm "$ZIP_NAME"

    echo "Removing unwanted files and folders..."
    rm -rf \
        "$TARGET_DIR/__MACOSX" \
        "$TARGET_DIR/spider_data/test_database" \
        "$TARGET_DIR/spider_data/dev_gold.sql" \
        "$TARGET_DIR/spider_data/test_gold.sql" \
        "$TARGET_DIR/spider_data/test_tables.json" \
        "$TARGET_DIR/spider_data/test.json" \
        "$TARGET_DIR/spider_data/train_gold.sql" \
        "$TARGET_DIR/spider_data/train_others.json"

    mv "$TARGET_DIR/spider_data"/* "$TARGET_DIR/"
    rm -rf "$TARGET_DIR/spider_data"

    # Convert JSON files to JSONL format
    echo "Converting JSON files to JSONL format..."

    JSON_FILES=("dev.json" "tables.json" "train_spider.json")

    for FILE in "${JSON_FILES[@]}"; do
        INPUT_PATH="$TARGET_DIR/$FILE"
        OUTPUT_PATH="${INPUT_PATH%.json}.jsonl"
        
        if [ -f "$INPUT_PATH" ]; then
            echo "Converting $FILE to JSONL..."
            jq -c '.[]' "$INPUT_PATH" > "$OUTPUT_PATH"
            rm "$INPUT_PATH"
        else
            echo "Warning: $INPUT_PATH not found. Skipping."
        fi
    done

    echo "Conversion complete. JSONL files are located in data/spider/"
else
    echo "Spider is already downloaded in data/spider/."
fi
