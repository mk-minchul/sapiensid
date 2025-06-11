#!/bin/bash

# Check if WEBBODY_DATA_ROOT is set
if [ -z "$WEBBODY_DATA_ROOT" ]; then
    echo "Error: WEBBODY_DATA_ROOT environment variable is not set"
    echo "Please set it to your WebBody data directory, e.g.:"
    echo "export WEBBODY_DATA_ROOT=/path/to/save/WebBody"
    exit 1
fi

if [[ "$WEBBODY_DATA_ROOT" =~ /$ ]]; then
    echo "Error: WEBBODY_DATA_ROOT must not end with /"
    exit 1
fi

# Create necessary directories
mkdir -p "$WEBBODY_DATA_ROOT/url_parquets"

# Parse command line arguments
DEBUG=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Download URL list
if [ "$DEBUG" = true ]; then
    python download_url.py --data_root "$WEBBODY_DATA_ROOT" --debug
    python make_parquet.py --input_file "$WEBBODY_DATA_ROOT/url.lst" \
                        --output_dir "$WEBBODY_DATA_ROOT/url_parquets" \
                        --num_parts 4
else
    python download_url.py --data_root "$WEBBODY_DATA_ROOT"
    python make_parquet.py --input_file "$WEBBODY_DATA_ROOT/url.lst" \
                        --output_dir "$WEBBODY_DATA_ROOT/url_parquets" \
                        --num_parts 49
fi
