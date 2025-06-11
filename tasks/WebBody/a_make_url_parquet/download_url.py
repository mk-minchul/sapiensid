import argparse
from huggingface_hub import hf_hub_download
import os

def download_url_list(data_root, debug=False):
    # Specify the repository and filename
    repo_id = "minchul/webbody_url_debug" if debug else "minchul/webbody_url"
    filename = "url.lst"

    # Download the file
    downloaded_file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=data_root  # Save to WEBBODY_DATA_ROOT
    )

    print(f"File downloaded to: {downloaded_file_path}")
    return downloaded_file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download WebBody URL list')
    parser.add_argument('--data_root', required=True, help='Root directory to save the data')
    parser.add_argument('--debug', action='store_true', help='Use debug dataset')
    args = parser.parse_args()

    download_url_list(args.data_root, args.debug)