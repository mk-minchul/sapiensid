# WebBody Data Processing

This repository contains scripts for processing WebBody dataset, including downloading URL lists and converting them to parquet format.

## Setup

1. Set the environment variable `WEBBODY_DATA_ROOT` to point to your WebBody data directory:
```bash
pip install -r requirements.txt
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cuml-cu12==24.8.*"
export WEBBODY_DATA_ROOT=/path/to/save/WebBody
export HF_TOKEN=YOUR_APITOKEN
wandb.login()
```



## Usage

### a. prepare URL

Depending on the size of the dataset you want to work with, you can use the following commands:
```bash
cd a_make_url_parquet
bash run.sh           # Use this if you want full url 
bash run.sh --debug   # Use this use smaller subset for testing
# subsequent steps will be same. They both download url.lst at WEBBODY_DATA_ROOT/ (but different in size).
# Above step determines the size of dataset.
```


### b. Download Images

This step downloads the images from the URLs and converts them to parquet format:
```bash
cd b_download
bash run.sh
```

The download process:
- Downloads images and converts them to parquet format using img2dataset
- Resizes images to 640x640 pixels
- Uses parallel processing for efficient downloads
- Saves additional columns including idx and label
- Outputs parquet files in the specified output directory

The script `run.sh` will create a `generated_script.sh` file that handles the actual downloading process. The download parameters are optimized for large-scale processing with:
- Multiple processes for parallel downloads
- Thread management for efficient resource usage
- Automatic retry on failures
- Quality control for image encoding


### c. Run YOLO Detection
Note: Make sure you have GPU for this step.

This step processes the downloaded images using YOLOv8 to detect and crop human bodies:

```bash
cd c_detect
bash run.sh --print_only  # do this if you have multiple GPUS to create script manually.
bash run.sh               # do this if you want to run it on one gpu sequentially
bash run.sh --delete      # this will delete the artifact from step c. 
```

The detection process:
- Uses YOLOv8-pose model to detect human bodies in images
- Processes images in parallel using multiple GPUs
- Saves detection results including:
  - Body keypoints (17 points per person)
  - Bounding boxes
  - Confidence scores
- Outputs are saved in `${WEBBODY_DATA_ROOT}/raw_img_parquets_640_yolo_cropped/`


### d. Crop face image

This step processes the YOLO-detected images to extract and align face regions:

```bash
cd d_crop
bash run.sh
```

The face cropping process:
- Uses RetinaFace model to detect and align faces in the upper body crops
- Saves outputs including:
  - Aligned face images
  - Facial landmark coordinates
  - Detection confidence scores
- Outputs are saved in `${WEBBODY_DATA_ROOT}/raw_img_parquets_640_yolo_cropped_retinaface_resnet50_aligned_cropsize_160_maxsize_320/`

The process includes quality control measures:
- Skips faces that are too small (bbox < 16 pixels)
- Saves visualization samples for monitoring
- Maintains original image metadata and labels


### e. Face recognition on cropped faces

Performs face feature extraction with vit_kprpe

```bash
cd e_facerec
bash run.sh  # take a look before running because multiGPU needs to be modified.
```

### f. cluster face features

This step performs clustering on the extracted face features to group similar faces together.

```bash
cd f_cluster
bash run.sh
```

### g. Clean and filter face clusters

This step performs final cleaning and filtering of the face clusters to ensure high-quality results:

```bash
cd g_clean
bash run.sh
```

The cleaning process:
- Filters faces based on quality scores (default threshold: 0.95)
- Merges similar clusters using a pre-clustering method
- Handles ambiguous cases with a threshold of 0.6
- Validates similarity with a threshold of 0.7
- Uses multiple GPUs for efficient processing
- Outputs cleaned and filtered face clusters in the cluster result directory

Key parameters:
- Face score filtering threshold: 0.95
- Merge threshold: 0.7
- Ambiguous threshold: 0.6

### i. Bundle Body Images

This step bundles the processed body images to a single rec file based on the face clustering results:

```bash
cd i_bundle_body
bash run.sh
```

### j. Split and Prepare Face-Body Dataset

This step splits and prepares the final webbody dataset:

```bash
cd j_make_split_face_body
python make_subset_config.py # must change according to your situation
bash build_subset.sh # must change according to your situation
```

`make_subset_config.py` will creat csv files for wb4m, wb12m, wb_testing subsets.
