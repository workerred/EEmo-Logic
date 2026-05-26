import matplotlib
matplotlib.use('TkAgg')
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_theme(
    font="Microsoft YaHei",
    rc={"axes.unicode_minus": False,},
    style="whitegrid"
)


# ==========================================
# 1. Core computation function
# ==========================================
def calculate_optimized_reward(vad_matrix, emb_matrix, alpha=0.6, k=2, p=3):
    """
    Compute the final Reward matrix for GRPO.

    Args:
        vad_matrix (list/np.array): VAD distance score matrix (0~1)
        emb_matrix (list/np.array): Raw embedding similarity matrix
        alpha (float): VAD weight (default 0.6)
        k (int): Embedding sharpening exponent (denoise, default 2)
        p (int): Final Reward peaking exponent (increase separation, default 3)
    """
    vad_mat = np.array(vad_matrix)
    emb_mat = np.array(emb_matrix)

    # --- A. Global Normalization ---
    # Goal: Within the current emotion group, find the strongest
    # non-self semantic association as the 1.0 benchmark.
    # This avoids the row-normalization issue where weak associations are forcibly inflated.

    # Create mask to exclude diagonal
    mask = ~np.eye(emb_mat.shape[0], dtype=bool)

    # Find the global maximum off-diagonal value
    if np.any(mask):
        global_max = np.max(emb_mat[mask])
    else:
        global_max = 1.0  # prevent division by zero for single-element matrices

    # Apply normalization
    if global_max > 0:
        emb_norm = emb_mat / global_max
    else:
        emb_norm = emb_mat

    # Set diagonal to 1.0 and clip to [0, 1]
    np.fill_diagonal(emb_norm, 1.0)
    emb_norm = np.clip(emb_norm, 0.0, 1.0)

    # --- B. Non-linear Sharpening ---
    # Goal: Suppress mid-low range noise (e.g., 0.4^2 = 0.16)
    emb_sharpened = np.power(emb_norm, k)

    # --- C. Weighted Fusion ---
    combined_score = alpha * vad_mat + (1 - alpha) * emb_sharpened

    # --- D. Final Peaking ---
    # Goal: Widen the gradient gap between correct answers (1.0) and suboptimal ones
    final_reward = np.power(combined_score, p)

    return final_reward


# Set plotting style
sns.set_theme(style="whitegrid")


def plot_overall_score_distribution(n, scores_matrix):
    """
    Plot the distribution of all values in a 2D score matrix.

    Args:
        n (int): Number of emotion options
        scores_matrix (list of list or np.array):
            2D matrix of shape (num_emotions, num_emotions).
            Example: [[0.1, 0.9], [0.8, 0.2], ...]
    """
    # 1. Data preprocessing: flatten the 2D list to a 1D array
    all_scores = np.array(scores_matrix).flatten()

    # 2. Create canvas
    plt.figure(figsize=(10, 6))

    # 3. Plot histogram with KDE (Kernel Density Estimation)
    # bins='auto' lets the algorithm choose an appropriate number of bins
    sns.histplot(all_scores, kde=True, bins=30, color='skyblue', edgecolor='black', alpha=0.7)

    # 4. Add statistical reference lines (optional)
    mean_val = np.mean(all_scores)
    median_val = np.median(all_scores)
    plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.3f}')
    plt.axvline(median_val, color='g', linestyle='-', label=f'Median: {median_val:.3f}')

    # 5. Set labels and title
    plt.title(f'Distribution of {n} Emotion Similarity Scores', fontsize=14)
    plt.xlabel('Similarity Score', fontsize=12)
    plt.ylabel('Frequency (Count)', fontsize=12)
    plt.legend()

    # 6. Display the chart
    plt.tight_layout()
    plt.show()


# ==========================================
# 2. Main processing pipeline
# ==========================================
def main():
    # File paths
    file_path_vad = 'vad_distance.json'
    file_path_emb = 'check_embedding.json'
    # Weighted fusion parameters
    alpha = 0.6
    k = 2
    p = 3
    # Output version
    version = 1

    # Load data
    try:
        with open(file_path_vad, 'r', encoding='utf-8') as f:
            vad_data_raw = json.load(f)
            # Handle possible list-of-dicts structure [{}, {}] -> {}
            vad_data = {}
            if isinstance(vad_data_raw, list):
                for item in vad_data_raw:
                    vad_data.update(item)
            else:
                vad_data = vad_data_raw

        with open(file_path_emb, 'r', encoding='utf-8') as f:
            emb_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: File not found {e.filename}")
        return

    # Results storage
    final_reward_dict = {}

    # Find emotion group keys common to both files
    common_keys = set(vad_data.keys()).intersection(set(emb_data.keys()))

    print(f"Found {len(common_keys)} emotion data groups to process...\n")

    for key in common_keys:
        labels = key.split(", ")
        n = len(labels)

        vad_mat = vad_data[key]
        emb_mat = emb_data[key]
        print(f"Processing: {key}")

        # Compute Reward
        # Parameters: alpha=0.6 (VAD-dominant), k=2 (embedding denoising), p=3 (strong separation)
        reward_mat_np = calculate_optimized_reward(vad_mat, emb_mat, alpha=alpha, k=k, p=p)

        # Convert to list for JSON serialization
        final_reward_dict[key] = reward_mat_np.tolist()

        # --- Analysis: Find max/min Rewards excluding 1.0 diagonal ---
        # Flatten all off-diagonal elements
        pairs = []
        for i in range(n):
            for j in range(n):
                if i != j:  # exclude self
                    pairs.append({
                        "target": labels[i],
                        "pred": labels[j],
                        "score": reward_mat_np[i, j]
                    })

        # Sort
        pairs.sort(key=lambda x: x['score'], reverse=True)

        # Print analysis results
        print("=" * 60)
        print(f"Emotion Group: {key[:50]}...")
        print("-" * 60)

        print("[Top 3 High Score Errors] (most confusing / high tolerance):")
        for item in pairs[:3]:
            print(f"  Target: {item['target']:<12} -> Pred: {item['pred']:<12} | Reward: {item['score']:.4f}")

        print("\n[Bottom 3 Low Score Penalties] (most severely penalized):")
        for item in pairs[-3:]:
            print(f"  Target: {item['target']:<12} -> Pred: {item['pred']:<12} | Reward: {item['score']:.4f}")
        print("\n")

    # ==========================================
    # 3. Save results
    # ==========================================
    output_filename = f'reward_matrix_{version}.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_reward_dict, f, indent=4, ensure_ascii=False)

    print(f"Processing complete! Final Reward dictionary saved to: {output_filename}")


if __name__ == "__main__":
    main()
