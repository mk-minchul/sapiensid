import os
import numpy as np
import matplotlib.pyplot as plt


def tensor_to_image(tensor):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    tensor = tensor.cpu().numpy()
    tensor = np.clip(tensor, 0, 50)

    cmap = cm.get_cmap('viridis', 51)  # Choose a gradual colormap like 'viridis'
    colors = cmap(np.arange(51))[:, :3]  # Exclude alpha channel
    image = colors[tensor]

    return image

def create_black_dot_image(h, w, i, j):
    import numpy as np
    # Initialize an array with ones (white pixels)
    image = np.ones((h, w, 3), dtype=np.float32)
    
    # Check if the provided indices are within the image bounds
    if 0 <= i < h and 0 <= j < w:
        # Set the specified pixel to black
        image[i, j] = [0, 0, 0]
    else:
        raise ValueError("Pixel indices (i, j) are out of bounds.")
    
    return image

def plot_bucket_ids(bucket_ids, max_height, max_width, skip=1):
    SAVE_DIR = 'kprpe_vis'
    os.makedirs(SAVE_DIR, exist_ok=True)
    bucket_ids_vis = bucket_ids.view(max_height, max_width, max_height, max_width)
    for i in range(max_height):
        for j in range(max_width):
            if i % skip != 0 or j % skip != 0:
                continue
            pos_vis = create_black_dot_image(max_height, max_width, i, j)
            bucket_vis = tensor_to_image(bucket_ids_vis[i, j])
            vis = np.concatenate([pos_vis, bucket_vis], axis=1)
            plt.imsave(f"{SAVE_DIR}/bucket_{i}_{j}.png", vis)