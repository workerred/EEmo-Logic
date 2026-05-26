# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications for emotion-aware GRPO training based on VideoChat-R1 framework.
# Sanitized for public release — no personal paths or internal references.

import logging
import os
import sys
import json
import re
import torch
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

import transformers
from transformers import set_seed
from accelerate import Accelerator, DistributedDataParallelKwargs
from typing import Optional

from tl_grpo.tr_grpo.trainer import GRPOConfig
from tl_grpo.tr_grpo import (
    Qwen2VLGRPOTrainer_Video_QA,
    Qwen2VLGRPOVLLMTrainer_Video_TG
)

from processing_qwen2_5_vl import Qwen2_5_VLProcessor
from modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from modeling_qwen2_5_vl_vllm import Qwen2_5_VLForConditionalGeneration_vllm
from my_qwen_utils import process_vision_info
from dataclasses import dataclass, field

# ============================================================
# Emotion category definitions for reward mapping
# ============================================================
# Use the 8-emotion set as default to support direct key lookup in reward_matrix_1.json.
# The 7-emotion set (Plutchik's basic emotions) can be enabled by setting use_seven=True.
emotion_8 = ['disgust', 'contentment', 'sadness', 'awe', 'anger', 'excitement', 'amusement', 'fear']
emotion_8_uppercase = [e.capitalize() for e in emotion_8]

# Core 7-emotion set (subset of 8-emotion set, excludes awe)
emotion_7 = ['disgust', 'contentment', 'sadness', 'anger', 'excitement', 'amusement', 'fear']
emotion_7_uppercase = [e.capitalize() for e in emotion_7]

# Helper: extract emotion label by removing parenthetical descriptions
def handle_label(input_str):
    """Clean the emotion label by stripping any appended description in parentheses."""
    if '(' in input_str:
        index = input_str.index('(')
        return input_str[:index].strip()
    return input_str.strip()

# ============================================================
# Reward Computation
# ============================================================
def get_reward(data_source, solution_str, ground_truth, extra_info=None, reward_database_path=None, use_seven=False):
    """
    Compute reward for a given solution.

    Supports two reward modes:
    1. Matrix-based reward (default for emotion_8): look up reward from reward_matrix_1.json
       based on ground_truth and predicted emotion.
    2. Binary reward (for emotion_7 / use_seven=True): exact match = 1.0, else 0.0
    """
    # Select emotion label list based on use_seven flag
    choice_list = emotion_7_uppercase if use_seven else emotion_8_uppercase

    # Parse the answer from the response string
    try:
        answer = solution_str.split(' response')[-1].strip().split(" ")[0]
    except:
        answer = "wrong"

    # Standardize the answer to match our label format
    answer = answer.strip().replace("_", " ").split(" (")[0]
    answer = handle_label(answer)

    # =====================================================
    # Reward Mode 1: Matrix-based (load reward_matrix_1.json)
    # Use when all options are within emotion_8 and use_seven=False
    # =====================================================
    if set(choice_list) == set(emotion_8_uppercase) and not use_seven:
        reward_database = json.load(open(reward_database_path, 'r', encoding='utf-8'))

        # Normalize the answer: map any case variation to the key format
        answer_normalized = answer.lower()
        if answer_normalized not in emotion_8:
            # Try to match by checking if the answer is a substring of any emotion in emotion_8
            for emo in emotion_8:
                if answer_normalized in emo or emo in answer_normalized:
                    answer_normalized = emo
                    break
            else:
                # Could not map -> penalty
                print(f"GRPO [WARNING] Could not map emotion '{answer}' to any in emotion_8. Awarding 0.")
                return 0.0

        # Construct the key format: "emotion1, emotion2, ..., emotion_n" (lowercase, sorted by order in emotion_8)
        gt_key_normalized = ground_truth.strip()

        # Try to find matching key in the reward database
        if gt_key_normalized not in reward_database:
            # Fallback: search for similar keys where all GT emotions are a subset
            found_key = None
            for db_key in reward_database.keys():
                gt_emotions = set(gt_key_normalized.lower().split(", "))
                db_emotions = set(db_key.lower().split(", "))
                if gt_emotions.issubset(db_emotions) and len(db_emotions) == len(emotion_8):
                    found_key = db_key
                    break
            if found_key:
                gt_key_normalized = found_key
            else:
                print(f"GRPO [WARNING] Could not find reward entry for key: {gt_key_normalized}. Awarding 0.")
                return 0.0

        reward_matrix = reward_database[gt_key_normalized]

        # Map the answer to its position in the emotion list
        gt_emotions_list = gt_key_normalized.split(", ")
        target_index = -1
        for i, emo in enumerate(gt_emotions_list):
            if emo.lower() == answer_normalized:
                target_index = i
                break
        if target_index == -1:
            print(f"GRPO [WARNING] Answer '{answer_normalized}' not in GT emotion list. Awarding 0.")
            return 0.0

        # Determine which emotion was the correct dominant one from extra_info
        if extra_info is not None:
            raw_answer = extra_info.get("raw_answer", "")
            if '(' in raw_answer:
                dominant_emotion = raw_answer.split('(')[0].strip().lower()
            else:
                dominant_emotion = raw_answer.strip().lower()
            gt_emotions_list_lower = [e.lower() for e in gt_emotions_list]
            try:
                row_idx = gt_emotions_list_lower.index(dominant_emotion)
                col_idx = gt_emotions_list_lower.index(answer_normalized)
                reward = float(reward_matrix[row_idx][col_idx])
            except (ValueError, IndexError):
                print(f"GRPO [WARNING] Could not find emotion indices. Awarding 0.")
                return 0.0
        else:
            # Without extra_info, assume the first emotion in the list is the correct dominant one
            gt_emotions_list_lower = [e.lower() for e in gt_emotions_list]
            try:
                row_idx = 0  # first emotion = dominant
                col_idx = gt_emotions_list_lower.index(answer_normalized)
                reward = float(reward_matrix[row_idx][col_idx])
            except (ValueError, IndexError):
                print(f"GRPO [WARNING] Could not find emotion index. Awarding 0.")
                return 0.0

        return reward

    # =====================================================
    # Reward Mode 2: Binary exact match (for emotion_7 / other tasks)
    # =====================================================
    else:
        # Parse predicted answer
        answer = solution_str.split(' response')[-1].strip().split(" ")[0]
        answer = answer.strip().replace("_", " ").split(" (")[0]
        answer = handle_label(answer)

        # Standardize punctuation in ground truth
        ground_truth_cleaned = handle_label(ground_truth)

        # Case-insensitive comparison
        if answer.lower() == ground_truth_cleaned.lower():
            return 1.0
        else:
            # Partial match handling for known variations
            if answer.lower() in ground_truth_cleaned.lower() or ground_truth_cleaned.lower() in answer.lower():
                if answer.lower() == "content" and ground_truth_cleaned.lower() == "contentment":
                    return 1.0
                if answer.lower() == "contentment" and ground_truth_cleaned.lower() == "content":
                    return 1.0
                if answer.lower() == "joy" and ground_truth_cleaned.lower() == "joyful":
                    return 1.0
                if answer.lower() == "joyful" and ground_truth_cleaned.lower() == "joy":
                    return 1.0
                return 0.0
            return 0.0

# ============================================================
# Dataset utilities
# ============================================================
def extract_emotion_options(question_text):
    """
    Extract the list of emotion options from a question following patterns:
    'Options: \nA. Emotion1\nB. Emotion2\n...' or 'Options: Emotion1, Emotion2, ...'
    Returns a list of emotion labels.
    """
    if "Options:" in question_text:
        options_part = question_text.split("Options:")[-1]
        # Try letter-prefixed format: A. Joy, B. Sad, etc.
        letter_matches = re.findall(r'[A-Z]\.\s*([A-Za-z]+(?:/[A-Za-z]+)?)', options_part)
        if letter_matches:
            filtered = [m.strip() for m in letter_matches if m.lower() not in [
                'contentment', 'joy', 'sadness', 'anger', 'fear', 'disgust', 'amusement', 'awe', 'excitement', 'neutral'
            ] or any(emo in m.lower() for emo in emotion_8)]
            if len(filtered) == 0:
                filtered = letter_matches
            return filtered
        # Pattern 2: Comma-separated after "Options:"
        options_text = options_part.strip()
        if ', ' in options_text:
            return [o.strip() for o in options_text.split(', ')]
        elif ',' in options_text:
            return [o.strip() for o in options_text.split(',')]
    return []

def extract_true_dominant_emotion(answer_text):
    """
    Extract the dominant (correct) emotion from the answer field.
    Format: "DominantEmotion(Option1, Option2, ...)" or just "DominantEmotion"
    """
    if '(' in answer_text:
        dominant = answer_text.split('(')[0].strip()
    else:
        dominant = answer_text.strip()
    return dominant

# ============================================================
# QA reward wrapper (Video QA trainer)
# ============================================================
def reward_func_qa(completions, assistant, **kwargs):
    """
    Reward function for video QA / emotion understanding tasks.
    Supports both matrix-based reward (from json) and binary reward modes.

    When the answer contains extra reasoning appended after the
    key answer word (in parenthesized detail), the handle_label helper strips it.
    """
    Rewards_list = []
    for i in range(len(completions)):
        data_source = ""
        solution_str = completions[i]
        ground_truth = assistant[i]

        # Extract reward mode and extra info from kwargs
        reward_database_path = kwargs.get("reward_database_path", "./reward_matrix_1.json")
        extra_info_dict = kwargs.get("extra_info", None)
        use_seven = kwargs.get("use_seven", False)

        # Call the common reward function
        try:
            if extra_info_dict is not None and i < len(extra_info_dict):
                extra = extra_info_dict[i]
            else:
                extra = None
            reward = get_reward(data_source, solution_str, ground_truth, extra, reward_database_path, use_seven)
        except Exception as e:
            print(f"GRPO [ERROR] reward computation failed: {e}")
            reward = 0.0

        Rewards_list.append(reward)

    return Rewards_list

# ============================================================
# TG reward wrapper (vLLM Video TG trainer)
# ============================================================
def reward_func(completions, assistant, **kwargs):
    """
    TG-style reward function (used with vLLM trainer).
    Same core logic, but adapted for TG trainer's calling convention.
    """
    Rewards_list = []
    for i in range(len(completions)):
        data_source = ""
        solution_str = completions[i]
        ground_truth = assistant[i]

        reward_database_path = kwargs.get("reward_database_path", "./reward_matrix_1.json")
        extra_info_dict = kwargs.get("extra_info", None)
        use_seven = kwargs.get("use_seven", False)

        try:
            if extra_info_dict is not None and i < len(extra_info_dict):
                extra = extra_info_dict[i]
            else:
                extra = None
            reward = get_reward(data_source, solution_str, ground_truth, extra, reward_database_path, use_seven)
        except Exception as e:
            print(f"GRPO [ERROR] reward computation failed: {e}")
            reward = 0.0

        Rewards_list.append(reward)

    return Rewards_list

# ============================================================
# Format reward
# ============================================================
def format_reward(completions, **kwargs):
    """
    Check if completions follow the expected format pattern and structure.
    Used to ensure model outputs conform to the required response format.

    Checks:
    1. Core pattern presence: reasoning + response (optionally with thinking tags).
    2. Emotion prediction is one of the valid values.
    """
    pattern = r"^<think>.*?</think>\s*response\s*[A-Za-z0-9\s\-/_\(\)]+$|^.*?\s*response\s*[A-Za-z0-9\s\-/_\(\)]+$"

    # Get valid emotions for comparison
    valid_emotions = {
        'disgust', 'content', 'contentment', 'sadness', 'sad', 'awe', 'anger',
        'annoyance', 'excitement', 'amusement', 'fear', 'pain', 'joy',
        'neutral', 'surprise', 'happiness'
    }

    rewards = []
    for completion in completions:
        completion_text = completion[0]["content"] if isinstance(completion, list) else completion
        # Check pattern match
        if re.match(pattern, completion_text, re.DOTALL):
            # Check for valid emotion word
            response_part = completion_text.split("response")[-1].strip()
            emotion_word = response_part.split()[0].strip('.,;:()[]{}').lower()
            if emotion_word in valid_emotions:
                reward_num = 1.0
            else:
                reward_num = 0.0
        else:
            reward_num = 0.0
        rewards.append(reward_num)
    return rewards

# ============================================================
# Main training arguments
# ============================================================
@dataclass
class GRPOScriptArguments(GRPOConfig):
    """
    Arguments for emotion GRPO training script.
    """
    dataset_name: str = field(default=None)
    dataset_config: str = field(default=None)
    reward_database_path: str = field(
        default="./reward_matrix_1.json",
        metadata={"help": "Path to the reward matrix JSON file."}
    )
    use_seven: bool = field(
        default=False,
        metadata={"help": "Use 7-emotion set (Plutchik's) instead of 8-emotion set."}
    )
    max_eval_samples: int = field(
        default=100,
        metadata={"help": "Maximum number of evaluation samples."}
    )
    reward_funcs: list = field(
        default_factory=lambda: ["format_reward", "reward_func_qa"],
        metadata={"help": "List of reward functions to use."}
    )

# ============================================================
# Main entry point
# ============================================================
def main(script_args, training_args, model_args):
    # Get reward configuration
    reward_database_path = script_args.reward_database_path
    use_seven = script_args.use_seven

    # Ensure the reward database exists when using matrix mode
    if not use_seven and not os.path.exists(reward_database_path):
        raise FileNotFoundError(
            f"Reward matrix file not found: {reward_database_path}. "
            "Run dominant_reward_dict.py first to generate it."
        )

    # Placeholder for full training loop setup
    # Configures model, processor, trainer, and launches training
    pass

if __name__ == "__main__":
    from tl_grpo.tr_grpo import HfArgumentParser
    parser = HfArgumentParser((GRPOScriptArguments, transformers.TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    main(script_args, training_args, None)
