#!/bin/bash

# Check if WEBBODY_DATA_ROOT is set
if [ -z "$WEBBODY_DATA_ROOT" ]; then
    echo "Error: WEBBODY_DATA_ROOT environment variable is not set"
    echo "Please set it using: export WEBBODY_DATA_ROOT=/path/to/save/WebBody"
    exit 1
fi

# Check if url.lst exists and get its size
URL_LST="${WEBBODY_DATA_ROOT}/url.lst"
if [ ! -f "$URL_LST" ]; then
    echo "Error: url.lst not found at ${URL_LST}"
    exit 1
fi

# Get file size in bytes and convert to MB
FILE_SIZE_KB=$(($(stat -c %s "$URL_LST") / 1024 ))
echo "URL File size: $FILE_SIZE_KB KB"

# Set num_parquets based on file size
if [ "$FILE_SIZE_KB" -lt 1000 ]; then
    echo "This is a Debug URL"
    NUM_PARQUETS=4
else
    echo "This is a Full URL"
    NUM_PARQUETS=49
fi

# full version
python make_download_script.py --output_dir "${WEBBODY_DATA_ROOT}/raw_img_parquets" \
                               --parquet_dir "${WEBBODY_DATA_ROOT}/url_parquets" \
                               --img_size 640 \
                               --num_parquets "$NUM_PARQUETS" \
                               --processes_count 89 \
                               --thread_count 128
bash generated_script.sh
