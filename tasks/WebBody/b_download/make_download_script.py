import os
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
import numpy as np
np.bool = np.bool_  # fix bug for mxnet 1.9.1
np.object = np.object_
import shutil
import os
import argparse
from general_utils.os_utils import natural_sort


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/localscratch/mckim/faces/webface260m/WebFaceV2_subset_by_face_score_v1/raw_img_parquets")
    parser.add_argument("--parquet_dir", type=str, default="/localscratch/mckim/faces/webface260m/WebFaceV2_subset_by_face_score_v1/url_parquets")
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument('--num_parquets', type=int, default=49)
    parser.add_argument('--processes_count', type=int, default=89)  # good to be number where all parquets are initalized together
    parser.add_argument('--thread_count', type=int, default=128)

    arg = parser.parse_args()

    # get all parquet files
    parquet_files = [os.path.join(arg.parquet_dir, f'{i+1}.parquet') for i in range(arg.num_parquets)]

    writer = open('generated_script.sh', 'w')
    # sort
    parquet_files = natural_sort(parquet_files)
    for idx, parquet_file in enumerate(parquet_files):
        img_size = arg.img_size
        output_dir = os.path.join(arg.output_dir+f'_{img_size}', os.path.basename(parquet_file).replace('.', '_'))

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        cmd_str = f"""
img2dataset --url_list {parquet_file} --input_format "parquet" \
--url_col "URL" --save_additional_columns '["idx", "label"]' --output_format "parquet" \
--output_folder {output_dir} --processes_count {arg.processes_count} --thread_count {arg.thread_count} \
--enable_wandb True --number_sample_per_shard 30000 \
--disallowed_header_directives '[]' --encode_quality 85 --image_size {img_size} \
--resize_only_if_bigger False --resize_mode border --timeout 20 --retries 1
        """
        # print(cmd_str)
        writer.write(cmd_str+'\n')

    writer.close()