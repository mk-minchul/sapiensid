import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
import os, sys
sys.path.append(os.path.join(root))
import numpy as np
np.bool = np.bool_  # fix bug for mxnet 1.9.1
np.object = np.object_
np.float = np.float_

from reader import SplittedFeatureRecordReader
from reader_img import SplittedRecordReader as SplittedRecordReaderImg
import argparse
from general_utils.os_utils import get_all_files
from general_utils.img_utils import tensor_to_pil, put_text_pil, stack_images
import cuml
import cupy as cp
from sklearn.cluster import DBSCAN as DBSCAN_CPU
import torch
import pandas as pd
import lovely_tensors as lt
lt.monkey_patch()
import math
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

# pip install \
#     --extra-index-url=https://pypi.nvidia.com \
#     "cuml-cu12==24.8.*"

def make_cluster_vis(cluster_labels, image_tensors, save_path):
    order = np.argsort(cluster_labels)
    vis_img_tensors = image_tensors[order]
    vis_cluster_labels = cluster_labels[order]
    pil_images = tensor_to_pil(vis_img_tensors)
    pil_images = [put_text_pil(pil, f'{lab}') for pil, lab in zip(pil_images, vis_cluster_labels)]
    n = len(pil_images)
    h = math.sqrt(n)
    w = math.ceil(n / h)
    h = math.ceil(h)
    vis = Image.fromarray(stack_images(pil_images, num_cols=h, num_rows=w).astype(np.uint8))
    vis.save(save_path)

def animate_dbscan_similarity(sim_mat, images, min_samples=2, eps_start=0.1, eps_end=1.0, eps_step=0.05,
                              gif_path='dbscan_similarity_animation.gif'):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from matplotlib.patches import Rectangle
    from PIL import Image

    """
    Function to animate DBSCAN clustering and visualize the similarity matrix, sorted by predicted labels,
    with images as x and y axis ticks.

    Args:
        sim_mat (numpy.ndarray): The similarity matrix (NxN).
        image_paths (list of str): The paths to the images that correspond to the features.
        min_samples (int): The minimum number of points to form a cluster.
        eps_start (float): The starting epsilon value.
        eps_end (float): The ending epsilon value.
        eps_step (float): The step size for epsilon increment.
        gif_path (str): File path to save the GIF.
    """
    # Convert similarity matrix to distance matrix (1 - similarity for cosine-like similarity)
    dist_mat = 1 - sim_mat

    # Load images and resize them to a uniform size (e.g., 30x30 pixels)
    side = int(30 * 15 / len(images))
    images = [image.resize((side, side)) for image in images]

    # Setup the figure
    fig, ax = plt.subplots(figsize=(16, 10))

    def sort_by_labels(matrix, labels):
        """Sort rows and columns of the matrix based on labels."""
        sorted_indices = np.argsort(labels)
        sorted_matrix = matrix[sorted_indices, :][:, sorted_indices]
        return sorted_matrix, sorted_indices

    def update_ticks(ax, sorted_indices):
        """Update x and y ticks with images based on sorted indices."""
        ax.set_xticks(np.arange(len(images)))
        ax.set_yticks(np.arange(len(images)))

        # Clear previous tick images
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add images as ticks slightly off from the matrix heatmap
        for i, img_idx in enumerate(sorted_indices):
            # For x-axis (place slightly above the matrix)
            img_x = OffsetImage(images[img_idx], zoom=1)
            ab_x = AnnotationBbox(img_x, (i, len(images) + 0.5), xycoords='data', frameon=False)  # Moved 0.5 units up
            ax.add_artist(ab_x)

            # For y-axis (place slightly to the left of the matrix)
            img_y = OffsetImage(images[img_idx], zoom=1)
            ab_y = AnnotationBbox(img_y, (-2, i), xycoords='data', frameon=False)  # Moved 2 units left
            ax.add_artist(ab_y)

        # Hide default tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Ensure that the limits of the axes are properly set to fit the images
        ax.set_xlim(-2.5, len(images))  # Added more room on the left for y-tick images
        ax.set_ylim(len(images), -1)

    def draw_cluster_boxes(ax, labels, sorted_indices):
        """Draw rectangles around the clusters formed by DBSCAN."""
        unique_labels = np.unique(labels)

        for label in unique_labels:
            if label == -1:
                # Skip noise points
                continue

            # Get sorted cluster indices
            cluster_indices = np.where(labels == label)[0]
            cluster_sorted_positions = np.isin(sorted_indices, cluster_indices).nonzero()[0]

            if len(cluster_sorted_positions) > 0:
                # The first and last index of this cluster in the sorted list
                first_idx = np.min(cluster_sorted_positions)
                last_idx = np.max(cluster_sorted_positions)

                # Create a rectangle for the cluster
                rect = Rectangle(
                    (first_idx - 0.5, first_idx - 0.5),  # Bottom left corner
                    last_idx - first_idx + 1,  # Width
                    last_idx - first_idx + 1,  # Height
                    fill=False, edgecolor='red', linewidth=2
                )
                ax.add_patch(rect)

    def animate(i):
        """Update the animation for each frame."""
        current_eps = eps_start + i * eps_step  # Update epsilon
        if current_eps > eps_end:
            current_eps = eps_end  # Keep the last frame for extra iterations

        # Apply DBSCAN with current epsilon
        dbscan_temp = DBSCAN(eps=current_eps, min_samples=min_samples, metric='precomputed')
        labels_temp = dbscan_temp.fit_predict(dist_mat)

        # Check if any clusters were found, if not skip the sorting
        if len(set(labels_temp)) == 1 and -1 in labels_temp:
            # Only noise was found
            sorted_sim_mat = sim_mat
            sorted_indices = np.arange(len(images))  # No sorting
        else:
            # Sort the similarity matrix based on labels
            sorted_sim_mat, sorted_indices = sort_by_labels(sim_mat, labels_temp)

        # Clear the axis and plot the sorted similarity matrix
        ax.clear()
        im = ax.imshow(sorted_sim_mat, cmap='jet', interpolation='nearest')
        ax.set_title(f'DBSCAN Clustering (Threshold={1-current_eps:.2f})', fontsize=32)
        ax.set_xlabel('$N \\times N$ Similarity Matrix', fontsize=18)

        # Update tick images
        update_ticks(ax, sorted_indices)

        # Draw the cluster boxes
        draw_cluster_boxes(ax, labels_temp, sorted_indices)

        # Remove the previous line and add the updated line
        if len(cbar.ax.lines) > 0:
            cbar.ax.lines[0].remove()

        # Normalize eps for colorbar range and add a new horizontal line
        eps_normalized = (current_eps - eps_start) / (eps_end - eps_start)
        eps_normalized = 1 - eps_normalized
        cbar.ax.hlines(eps_normalized * (cbar.vmax - cbar.vmin) + cbar.vmin, *cbar.ax.get_xlim(), color='red', linewidth=2)

    # Initialize the colorbar outside the animate function to avoid flickering
    im = ax.imshow(sim_mat, cmap='jet', interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, orientation='vertical')

    # Increase the fontsize of the colorbar label
    cbar.ax.set_ylabel('Similarity', fontsize=18)

    # Calculate the total number of frames, including 3 extra for the final state
    num_frames = int((eps_end - eps_start) / eps_step) + 5

    # Create the animation
    anim = FuncAnimation(fig, animate, frames=num_frames, interval=500)

    # Save the animation as a GIF
    writer = PillowWriter(fps=2)  # Set fps for animation speed
    anim.save(gif_path, writer=writer)

    return gif_path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--record_root', type=str, default='/ssd2/data/faces/webface260m/WebFaceV2_subset_by_face_score_v1/face_features/vit_base_kprpe_webface12m/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320')

    parser.add_argument('--eps', type=float, default=0.7)
    parser.add_argument('--min_samples', type=int, default=2)

    args = parser.parse_args()

    record_paths = get_all_files(args.record_root, extension_list=['.rec'], sort=True)
    print('Found records:', len(record_paths))
    record_paths = [os.path.dirname(path) for path in record_paths]
    dataset = SplittedFeatureRecordReader(record_paths)

    sample, label, path = dataset.read_by_index(1)

    img_record_path = '/hdd3/data/faces/webface260m/WebFaceV2_subset_by_face_score_v1/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320/3_parquet'
    img_reader = SplittedRecordReaderImg([img_record_path])
    transform = Compose([ToTensor(),
                         Resize((112, 112), antialias=True),
                         Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])

    df = pd.DataFrame(pd.Series(dataset.paths), columns=['path'])
    df['label'] = df['path'].apply(lambda x: x.split('/')[0])
    groupby = df.groupby('label')
    for label, group in groupby:
        print(label, len(group))
        if len(group) <= 1:
            continue
        feats = torch.stack([dataset.read_by_path(path)[0] for path in group['path']])
        feats = feats.to('cuda')
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        similarity_matrix_cuda = torch.mm(feats, feats.T)
        dbscan_cuda = cuml.DBSCAN(eps=args.eps, min_samples=args.min_samples, metric='precomputed')
        cluster_labels_cuda = dbscan_cuda.fit_predict(1 - similarity_matrix_cuda)

        # higher eps leads to more samples being same subject I changed from 0.5 to 0.7
        # min_samples is the minimum number of samples in a cluster
        similarity_matrix_np = similarity_matrix_cuda.cpu().detach().numpy()
        similarity_matrix_np = np.clip(similarity_matrix_np, 0, 1)
        dbscan_cpu = DBSCAN_CPU(metric='precomputed', eps=args.eps, min_samples=args.min_samples)
        cluster_labels_cpu = dbscan_cpu.fit_predict(1 - similarity_matrix_np)

        img_paths = group['path'].values
        image_tensors = torch.stack([transform(img_reader.read_by_path(img_path)[0]) for img_path in img_paths])

        if len(np.unique(cluster_labels_cuda)) > 2 and len(image_tensors) > 4:

            similarity_matrix_np[np.arange(len(similarity_matrix_np)), np.arange(len(similarity_matrix_np))] = 0
            from general_utils.img_utils import tensor_to_pil
            images = tensor_to_pil(image_tensors)
            animate_dbscan_similarity(similarity_matrix_np, images, min_samples=2, eps_start=0.2, eps_end=0.7, eps_step=0.05,
                                      gif_path=f'/mckim/temp/cluster_vis/{label}_similarity.gif')
            # make_cluster_vis(cluster_labels_cpu, image_tensors, f'/mckim/temp/cluster_vis/{label}_1.png')
            make_cluster_vis(np.array(cluster_labels_cuda.get()), image_tensors, f'/mckim/temp/cluster_vis/{label}_2.png')

        # topk_cluster_label = most_frequent_non_negative_label(cluster_labels)


