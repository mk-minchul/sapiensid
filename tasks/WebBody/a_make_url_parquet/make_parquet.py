import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from tqdm import tqdm
from itertools import islice

def process_batch(urls, writer):
    """Convert a list of URLs to a PyArrow Table and append to a Parquet file."""
    if urls:
        df = pd.DataFrame(urls, columns=['idx', 'label', 'URL'])
        table = pa.Table.from_pandas(df)
        writer.write_table(table)

def convert_urls_to_parquet(input_file, output_file, batch_size=10000, part=0, num_parts=10, total_lines=None):
    """Read URLs from a text file in batches, divide into parts, and write to Parquet files."""
    urls = []

    part_size = total_lines // num_parts
    start = part * part_size
    end = start + part_size if part < num_parts - 1 else total_lines

    with open(input_file, 'r') as file:
        first_batch = True
        writer = None
        lines_to_process = end - start

        print(f"Processing part {part+1} of {num_parts} with {lines_to_process} lines from {start} to {end}.")

        # Using islice to jump directly to start line and only iterate until end line
        for line in tqdm(islice(file, start, end), desc=f"Reading URLs part {part+1}", total=lines_to_process):
            parts = line.strip().split()
            idx = parts[0]
            label = parts[1]
            url = " ".join(parts[2:])
            urls.append([idx, label, url])

            if len(urls) == batch_size:
                if first_batch:
                    print(urls[0])
                    writer = pq.ParquetWriter(output_file,
                                              pa.Table.from_pandas(pd.DataFrame(urls, columns=['idx', 'label', 'URL'])).schema,
                                              compression='snappy', use_dictionary=True)
                    first_batch = False
                process_batch(urls, writer)
                urls.clear()
                if debug:
                    break

        if urls:  # Handle the last batch if it exists
            if first_batch:
                writer = pq.ParquetWriter(output_file, pa.Table.from_pandas(pd.DataFrame(urls, columns=['idx', 'label', 'URL'])).schema,
                                          compression='snappy', use_dictionary=True)
            process_batch(urls, writer)

        if writer:
            writer.close()
        print(f"Parquet file part {part+1} created successfully.")


def count_rows_parquet(file_path):
    """Count the number of rows in a Parquet file using PyArrow."""
    table = pq.read_table(file_path)
    # Count  rows
    return table.num_rows


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='/ssd2/data/faces/webface260m/WebFaceV2/url.lst')
    parser.add_argument("--output_dir", type=str, default='/ssd2/data/faces/webface260m/WebFaceV2/url_parquets')
    parser.add_argument("--num_parts", type=int, default=49)
    args = parser.parse_args()

    # Define file paths and batch size
    input_file = args.input_file
    output_dir = args.output_dir
    num_parts = args.num_parts

    debug = False
    batch_size = 10000
    if debug:
        output_dir = output_dir.replace('_parquet', '_parquet_debug')
    os.makedirs(output_dir, exist_ok=True)

    total_lines = sum(1 for line in open(input_file, 'r'))
    nrows = 0
    for part in range(num_parts):
        output_file = f"{output_dir}/{part+1}.parquet"
        convert_urls_to_parquet(input_file, output_file, batch_size, part, num_parts, total_lines)
        nrows += count_rows_parquet(output_file)
    print(f"Total number of rows in Parquet files: {nrows}")
    print(f'Original number of lines in the input file: {total_lines}')
