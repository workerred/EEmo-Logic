import numpy as np
import json


def get_row_normalized_matrix_with_extremes(input_keys_str, weights=(0.5, 0.3, 0.2)):
    keys = [k.strip().lower() for k in input_keys_str.split(',')]  # must be lowercase
    n = len(keys)

    # Get VAD vectors
    vectors = []
    for k in keys:
        if k in vad_data:
            vectors.append(vad_data[k])
        else:
            print(f"Warning: {k} missing, using default.")
            vectors.append([5.0, 5.0, 5.0])
    vectors = np.array(vectors)

    w_v, w_a, w_d = weights
    final_matrix = []

    # For tracking extremes
    # 1. Highest off-diagonal Reward (Most Similar but not same)
    max_reward_val = -1.0
    max_reward_pair = {}

    # 2. Farthest 0.0 Reward (largest physical distance)
    max_physical_distance = -1.0
    min_reward_pair_info = {}

    # Iterate over each row (Target Emotion)
    for i in range(n):
        row_distances = []
        # Compute physical distance from this row to all other points
        for j in range(n):
            diff = vectors[i] - vectors[j]
            # Weighted Euclidean distance
            dist = np.sqrt(w_v * diff[0] ** 2 + w_a * diff[1] ** 2 + w_d * diff[2] ** 2)
            row_distances.append(dist)

        # Find the maximum physical distance in this row (used as normalization denominator)
        max_row_dist = max(row_distances)

        # Track the global maximum physical distance (most extreme opposition)
        # The corresponding Reward will definitely be 0.0
        if max_row_dist > max_physical_distance:
            max_physical_distance = max_row_dist
            farthest_idx = row_distances.index(max_row_dist)
            min_reward_pair_info = {
                "target": keys[i],
                "predicted": keys[farthest_idx],
                "reward": 0.0,
                "raw_distance": max_row_dist
            }

        row_scores = []
        for j, dist in enumerate(row_distances):
            # Row normalization
            if max_row_dist == 0:
                score = 1.0
            else:
                score = 1.0 - (dist / max_row_dist)

            row_scores.append(score)

            # Track the highest Reward (excluding diagonal i != j)
            if i != j:
                if score > max_reward_val:
                    max_reward_val = score
                    max_reward_pair = {
                        "target": keys[i],
                        "predicted": keys[j],
                        "reward": score,
                        "raw_distance": dist
                    }

        final_matrix.append(row_scores)

    result_dict = {input_keys_str: final_matrix}

    return result_dict, max_reward_pair, min_reward_pair_info


# ==========================================
# Example execution
# ==========================================

# 1. Load the VAD data prepared earlier
input_json = "emotion_vad_scores.json"
output_json = "vad_distance.json"
with open(input_json, "r", encoding='utf-8') as f:
    vad_data = json.load(f)
result = []  # record final results
option_str_list = ['disgust, contentment, sadness, awe, anger, excitement, amusement, fear',
                   'Anger, Sad, Fear, Disgust, Amusement, Excitement, Awe, Content',
                   'Pain, Sadness, Embarrassment, Fatigue, Excitement, Engagement, Confidence, Aversion, Happiness, Yearning, Affection, Sensitivity, Fear, Sympathy, Suffering, Doubt/Confusion, Disapproval, Anticipation, Anger, Esteem, Surprise, Disquietment, Pleasure, Disconnection, Peace, Annoyance',
                   'Trust, Admiration, Distraction, Joy, Boredom, Terror, Amazement, Disgust, Rage, Vigilance, Anticipation, Anger, Loathing, Grief, Acceptance, Fear, Pensiveness, Serenity, Surprise, Interest, Sadness, Apprehension, Annoyance, Ecstasy',
                   'fear, joy, anger, sadness, surprise, disgust, neutral']
for input_keys in option_str_list:
    # Set weights: Valence dominant (0.5), Arousal secondary (0.3), Dominance auxiliary (0.2)
    weights = (0.5, 0.3, 0.2)

    matrix_dict, high_reward, low_reward = get_row_normalized_matrix_with_extremes(input_keys, weights)
    result.append(matrix_dict)

    print("\n--- Reward Extremes Analysis ---")
    print(f"1. Highest Non-Self Reward:")
    print(f"   Description: This is the most confusable error with the lightest penalty.")
    print(f"   Target: '{high_reward['target']}'")
    print(f"   Predicted: '{high_reward['predicted']}'")
    print(f"   Reward Score: {high_reward['reward']:.6f} (close to 1.0)")
    print(f"   (Reason: They are extremely close in VAD space)")

    print(f"\n2. Lowest Reward with Max Difference:")
    print(f"   Description: Among pairs with Reward 0.0, this has the largest physical distance (strongest opposition).")
    print(f"   Target: '{low_reward['target']}'")
    print(f"   Predicted: '{low_reward['predicted']}'")
    print(f"   Reward Score: {low_reward['reward']:.1f}")
    print(f"   (Reason: Raw weighted distance is {low_reward['raw_distance']:.4f}, the largest across all pairs)")

# Print results
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=4)
