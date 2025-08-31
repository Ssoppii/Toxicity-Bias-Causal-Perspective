import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "axes.unicode_minus": False
})
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import argparse
from glob import glob
import pickle
from itertools import zip_longest

# ---- CONFIG ----
EMBEDDING_PATH = "glove.840B.300d.txt"  # Path to your GloVe embedding
CATEGORY_FILES = []

# ---- LOAD EMBEDDINGS ----
def load_glove_embeddings(glove_path):
    cache_path = glove_path[:-4] + ".pkl"
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    embeddings = {}
    print("Reading GloVe vectors from text...")
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 301:
                continue  # Skip lines that are too short
            word = parts[0]
            try:
                vec = np.array(parts[1:], dtype=np.float32)
                if vec.shape[0] != 300:
                    continue
                embeddings[word] = vec
            except ValueError:
                continue  # Skip lines with invalid float conversion

    with open(cache_path, 'wb') as f:
        pickle.dump(embeddings, f)
        print(f"Saved cached embeddings to {cache_path}")
    return embeddings

# ---- MAIN ----
def visualize_defining_sets(embedding_path, defining_files):
    print("Loading embeddings...")
    embeddings = load_glove_embeddings(embedding_path)

    plane_normals = []
    plane_midpoints = []
    colors = ['purple', 'green', 'blue', 'orange']

    for file in defining_files:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        category_name = Path(file).stem.replace("defining_set_", "")
        
        with open(file, "r", encoding="utf-8") as f:
            defining_data = json.load(f)
        
        analogy_templates = defining_data.get("analogy_templates", {}).get("role", {})
        category_labels = defining_data.get("category_labels", {})
        group_a_label = category_labels.get("positive_label", "group_a")
        group_b_label = category_labels.get("negative_label", "group_b")
        group_a_words = analogy_templates.get(group_a_label, [])
        group_b_words = analogy_templates.get(group_b_label, [])

        vecs_1, vecs_2 = [], []
        labels = []
        for w1, w2 in zip_longest(group_a_words, group_b_words, fillvalue=None):
            if w1 is None or w2 is None:
                continue
            if w1 not in embeddings or w2 not in embeddings:
                print(f"[{category_name}] Skipped: ({w1}, {w2}) - missing in embeddings")
                continue
            vecs_1.append(embeddings[w1])
            vecs_2.append(embeddings[w2])
            labels.append(f"{w1}-{w2}")

        if not vecs_1 or not vecs_2:
            print(f"[{category_name}] No valid pairs found.")
            continue

        vecs_1 = np.stack(vecs_1)
        vecs_2 = np.stack(vecs_2)

        pca = PCA(n_components=3)
        all_vecs = np.vstack([vecs_1, vecs_2])
        reduced_all = pca.fit_transform(all_vecs)
        reduced_1 = reduced_all[:len(vecs_1)]
        reduced_2 = reduced_all[len(vecs_1):]

        # Concatenate points for axis limits
        all_points = np.vstack([reduced_1, reduced_2])
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

        # Set axis limits with padding
        padding = 1.0
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_zlim(z_min - padding, z_max + padding)

        red_dot = ax.scatter(reduced_1[:, 0], reduced_1[:, 1], reduced_1[:, 2], color='red', s=40, alpha=0.7, label=f"{group_b_label}")
        blue_dot = ax.scatter(reduced_2[:, 0], reduced_2[:, 1], reduced_2[:, 2], color='blue', s=40, alpha=0.7, label=f"{group_a_label}")

        for i in range(len(reduced_1)):
            ax.plot([reduced_1[i, 0], reduced_2[i, 0]],
                    [reduced_1[i, 1], reduced_2[i, 1]],
                    [reduced_1[i, 2], reduced_2[i, 2]],
                    linestyle='dotted', color='gray', alpha=0.5)

        # Create 3D bias plane using bias direction and two orthogonal vectors
        line_scale_3d = 4
        origin = np.mean((reduced_1 + reduced_2) / 2, axis=0)

        # Normalize bias direction and find two orthogonal vectors
        bias_direction_3d = np.mean(vecs_2 - vecs_1, axis=0)
        normal = bias_direction_3d / np.linalg.norm(bias_direction_3d)
        if np.allclose(normal[:3], [1, 0, 0]):
            ref_vec = np.array([0, 1, 0])
        else:
            ref_vec = np.array([1, 0, 0])

        u = np.cross(normal[:3], ref_vec)
        u /= np.linalg.norm(u)
        v = np.cross(normal[:3], u)
        v /= np.linalg.norm(v)

        # Create grid
        grid_range = np.linspace(-line_scale_3d, line_scale_3d, 10)
        uu, vv = np.meshgrid(grid_range, grid_range)

        # Parametrize plane
        xx = origin[0] + uu * u[0] + vv * v[0]
        yy = origin[1] + uu * u[1] + vv * v[1]
        zz = origin[2] + uu * u[2] + vv * v[2]

        plane_normals.append(normal)

        ax.set_title(f"3D Bias Subspace Visualization - {category_name.upper()}", fontsize=20)
        ax.view_init(elev=30, azim=45)
        ax.set_box_aspect([1, 1, 1])

        plane = ax.plot_surface(xx, yy, zz, alpha=0.3, color=colors[len(plane_normals) % len(colors)], label=f"{category_name} bias plane")
        ax.legend(handles=[red_dot, blue_dot, plane], fontsize=14)

        # Highlight points close to the bias plane
        # threshold = 0.5  # Define a threshold for proximity to the plane
        # mean_midpoint_reduced = np.mean((reduced_1 + reduced_2) / 2, axis=0)
        # for point in reduced_all:
        #     distance = abs(np.dot(normal[:3], point - mean_midpoint_reduced))
        #     if distance < threshold:
        #         ax.scatter(point[0], point[1], point[2], color='yellow', s=50, alpha=0.8)

        plt.tight_layout()
        save_path = os.path.join(SAVE_DIR, f"{category_name}_subspace_3d.png")
        plt.savefig(save_path)
        print(f"[{category_name}] Saved 3D visualization to {save_path}")
        plt.close()

        # 2D Plot using PC1 and PC2
        _, ax2d = plt.subplots(figsize=(8, 6))

        # Normalize bias direction
        bias_vectors_2d = reduced_2 - reduced_1
        bias_direction_2d = np.mean(bias_vectors_2d, axis=0)

        for vec1, vec2 in zip(reduced_1, reduced_2):
            ax2d.scatter(vec1[0], vec1[1], color='red', s=30)   # group_b
            ax2d.scatter(vec2[0], vec2[1], color='blue', s=30)  # group_a
            ax2d.plot([vec1[0], vec2[0]], [vec1[1], vec2[1]], linestyle='dotted', color='gray', alpha=0.6)

        # Bias direction (as average vector from group A to B)
        origin = np.mean((reduced_1 + reduced_2) / 2, axis=0)
        line_scale = 10
        ax2d.plot(
            [origin[0] - bias_direction_2d[0]*line_scale, origin[0] + bias_direction_2d[0]*line_scale],
            [origin[1] - bias_direction_2d[1]*line_scale, origin[1] + bias_direction_2d[1]*line_scale],
            color='black', linestyle='--', label=f"{category_name} bias direction"
        )

        ax2d.set_title(f"2D Bias Subspace Visualization - {category_name.upper()}", fontsize=14)

        custom_lines = [
            Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=6, label=group_b_label),
            Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=6, label=group_a_label),
            Line2D([0], [0], color='black', linestyle='--', label=f"{category_name} bias direction")
        ]
        ax2d.legend(handles=custom_lines)

        plt.tight_layout()
        save_path_2d = os.path.join(SAVE_DIR, f"{category_name}_subspace_2d.png")
        plt.savefig(save_path_2d)
        print(f"[{category_name}] Saved 2D visualization to {save_path_2d}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to directory containing defining_set_*.json")
    parser.add_argument("--output_dir", type=str, default="../plots/bias_subspace_plots", help="Directory to save output plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    CATEGORY_FILES = sorted(glob(os.path.join(args.input_dir, "defining_set_*.json")))
    SAVE_DIR = args.output_dir

    visualize_defining_sets(EMBEDDING_PATH, CATEGORY_FILES)