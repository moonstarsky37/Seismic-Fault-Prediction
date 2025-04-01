import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List

def save_images(data: np.ndarray, predictions: np.ndarray, output_dir: str, slices: List[int]) -> None:
    """
    Saves original seismic slices, predicted fault slices, and overlay images to the specified directory.
    Args:
        data (np.ndarray): The normalized seismic data.
        predictions (np.ndarray): The fault prediction data.
        output_dir (str): Directory to save the images.
        slices (List[int]): List of slice indices to visualize.
    """
    os.makedirs(output_dir, exist_ok=True)
    for k in slices:
        orig_slice = data[k, :, :]
        pred_slice = predictions[k, :, :]

        # Save original seismic slice
        plt.figure(figsize=(6, 6))
        plt.title(f"Seismic Slice {k}")
        plt.imshow(orig_slice, cmap="gray")
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, f"slice_{k}_seismic.png"))
        plt.close()

        # Save predicted fault slice
        plt.figure(figsize=(6, 6))
        plt.title(f"Predicted Fault Slice {k}")
        plt.imshow(pred_slice, cmap="bone")
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, f"slice_{k}_prediction.png"))
        plt.close()

        # Save overlay image
        plt.figure(figsize=(6, 6))
        plt.title(f"Overlay Prediction {k}")
        plt.imshow(orig_slice, cmap="gray")
        plt.imshow(pred_slice, cmap="Reds", alpha=0.4)
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, f"slice_{k}_overlay.png"))
        plt.close()
    print(f"All predicted images have been saved to: {output_dir}")