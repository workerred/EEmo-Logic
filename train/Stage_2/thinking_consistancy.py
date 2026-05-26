# Requires transformers>=4.51.0
import json
import os

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import re
from scipy.stats import kendalltau


# Approximately 27GB (dual GPU)
def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'


def _extract_answer_content(text: str) -> str:
    """
    Helper function: Strip <think>...</think> from model output.
    If no tags present, return the entire text.
    """
    answer_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()

    # No tags found, return entire text
    return text.strip()


# Determine the most appropriate emotion via embedding matching
def embedding_judge(instruct, description, documents, response, task):
    queries = [
        get_detailed_instruct(instruct, description),
    ]
    keywords_list = list(documents.keys())
    doc_list = list(documents.values())  # keyword + definition description
    input_texts = queries + doc_list
    # Tokenize the input texts
    batch_dict = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch_dict.to(model.device)
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:1] @ embeddings[1:].T).tolist()[0]
    if task == "Valence":
        scores[1] = scores[1] - 0.05
    print(scores)
    score_dict = {keywords_list[i]: scores[i] for i in range(len(scores))}
    score_dict = dict(sorted(score_dict.items(), key=lambda x: x[1], reverse=True))  # sort by similarity descending
    print(f"({task})score_dict: {score_dict}")
    # Convert to numpy array
    data_array = np.array(scores)
    # Determine based on response type
    if type(response) == str or type(response) == float:
        # VAD and dominant emotion cases
        max_index = np.argmax(data_array)
        final_state = keywords_list[max_index]
    elif type(response) == list:
        # ranking case
        n = len(response)  # number of predicted emotions
        top_n_indices = np.argsort(data_array)[-n:][::-1].tolist()  # top-n by similarity descending
        final_state = [keywords_list[i] for i in top_n_indices]
    print(f"({task})final_state: {final_state}")
    return final_state, str(score_dict)


# Compare reasoning meaning with output response relevance
def similarity_score(final_state, response, task_id):
    # VAD tasks
    if task_id in [0, 1, 2]:
        if response <= 0.4:
            response_state = "low"
        elif response >= 0.6:
            response_state = "high"
        else:
            response_state = "moderate"
        # Same level
        if final_state == response_state:
            score = 1
        else:
            score = 0
    # ranking
    elif task_id == 3:
        ground_truth = response  # model output as GT
        # deduplicate
        g_t, p_emo = exclude_emo(ground_truth, final_state)
        if len(g_t) < len(p_emo):
            p_emo = p_emo[:len(g_t)]
        elif len(g_t) > len(p_emo):
            g_t = g_t[:len(p_emo)]
        score = float((hit_score(ground_truth, final_state) + kendall_tau(g_t, p_emo)) / 100)  # normalize
        # post-processing
        score = score ** 2
    # dominant emotion
    else:
        if final_state.lower() == response.lower():
            score = 1
        else:
            score = 0
    return score


# Compute Kendall's tau
def kendall_tau(ground_truth, predict):
    if not predict:  # empty
        score = 0
        return score
    tau, p_value = kendalltau(ground_truth, predict)
    # Scale from [-1, 1] to [0, 50]
    score = (tau + 1) * 25
    # Only 1 shared element
    if len(predict) == 1:
        score = 0
    # 2 shared elements
    elif len(predict) == 2:
        score = score * 2 / 3
    score = round(score, 2)
    return score


# Compute hit rate
def hit_score(ground_truth, predict):
    score = 0
    # hit weights
    weight = [25, 15, 10]
    for i in range(len(ground_truth)):
        if ground_truth[i] in predict:
            score += weight[i]
    return score


def exclude_emo(ground_truth, predict):
    new_predict = []
    new_ground_truth = []
    # Filter predicted emotions
    for emo in predict:
        emo = emo.replace("Neutrality", "Neutral")
        emo = emo.replace("Joyful", "Joy")
        emo = emo.replace("Shock", "Surprise")
        emo = emo.replace("Pleasure", "Joy")
        # Emotion in GT and not duplicate
        if emo in ground_truth and emo not in new_predict:
            new_predict.append(emo)
    # Match ground truth format
    for emo in ground_truth:
        # Skip emotions not present
        if emo in new_predict:
            new_ground_truth.append(emo)
    return new_ground_truth, new_predict


# -------------------------------------------------------------------------
scale = "8B"  # ~38G + 41G, dual GPU
model_dir = f"path/to/Qwen3-Embedding-{scale}"
tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side='left')
# We recommend enabling flash_attention_2 for better acceleration and memory saving.
model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto")
max_length = 8192

# Fixed prompt templates (5 tasks)
prompt_template = [
    (r"Given a paragraph describing the emotional valence evoked by an image, match it to the corresponding valence level (Low, Moderate, or High). "
     r"Focus on the degree of positivity or negativity described."),
    (r"Given a paragraph describing the emotional arousal evoked by an image, match it to the corresponding arousal level (Low, Moderate, or High). "
        r"Focus on the degree of calmness or intensity described."),
    (r"Given a paragraph describing the emotional dominance evoked by an image, match it to the corresponding dominance level (Low, Moderate, or High). "
        r"Focus on the degree of helplessness or powerfulness described."),
    (r"Given the following paragraph describing the emotions that may be evoked by a image, "
    r"choose the most relevant keyword or phrase that best summarize the emotions evoked. Retrieve word or phrase that capture the overall mood, feelings, or emotional response."),
    (r"Given the following paragraph describing the emotions that may be evoked by a image, "
     r"choose the most relevant keyword or phrase that best summarize the emotions evoked. Retrieve word or phrase that capture the overall mood, feelings, or emotional response."),
]

# Matching keywords or level keywords with definition descriptions
anchors = {
    "Valence": {
                "high": "Level: High. The emotion is extremely pleasant, positive, joyful, happy, and satisfying.",
                "moderate": "Level: Moderate. The emotion is neutral, ordinary, or mixed, with no strong positive or negative feelings.",
                "low": "Level: Low. The emotion is unpleasant, negative, sad, miserable, unhappy, or annoying."
            },
    "Arousal": {
        "high": "Level: High. The emotion is intense, excited, stimulated, frantic, jittery, or wide-awake.",
        "moderate": "Level: Moderate. The emotion is alert and awake but not intense; a normal state of attention.",
        "low": "Level: Low. The emotion is calm, relaxed, sluggish, sleepy, bored, or peaceful."
    },
    "Dominance": {
        "high": "Level: High. The observer feels dominant, in control, influential, important, confident, or bold.",
        "moderate": "Level: Moderate. The observer feels a balanced sense of control; neither overwhelmed nor commanding.",
        "low": "Level: Low. The observer feels submissive, influenced, awed, helpless, weak, or controlled by the situation."
    },
    "Emotion ranking": {
    # 1. Anger
    "anger": (
        "Emotion: Anger. "
        "The feeling of hostility, fury, frustration, and aggression. "
        "A strong negative reaction to provocation, injustice, or offense."
    ),
    # 2. Disgust
    "disgust": (
        "Emotion: Disgust. "
        "The feeling of intense revulsion, aversion, or sickness "
        "towards something offensive, toxic, contaminated, or morally wrong."
    ),
    # 3. Fear
    "fear": (
        "Emotion: Fear. "
        "The feeling of terror, panic, fright, or anxiety caused by imminent danger or threat. "
        "Unlike surprise, fear involves a sense of vulnerability and a desire to escape."
    ),
    # 4. Joy
    "joy": (
        "Emotion: Joy. "
        "The feeling of great pleasure, delight, ecstasy, and elation. "
        "A highly positive state of vibrant happiness and satisfaction, distinct from neutral surprise."
    ),
    # 5. Sadness
    "sadness": (
        "Emotion: Sadness. "
        "The feeling of sorrow, grief, misery, and depression. "
        "A low-energy state associated with loss, pain, or hopelessness."
    ),
    # 6. Surprise
    "surprise": (
        "Emotion: Surprise. "
        "A brief, neutral state of being startled by an unexpected event. "
        "It is a fleeting reflex that has not yet turned into fear or joy. "
        "Strictly neutral valence, purely reacting to the suddenness."
    ),
    # 7. Neutral
    "neutral": (
        "Emotion: Neutral. "
        "A calm, indifferent, and objective state. "
        "Lacking any strong emotional reaction, excitement, or arousal."
    )
},
    "Dominant emotion": {}  # specific samples feed emotion words directly
}


def find_last_generate(data):
    return 0


# JSON path for benchmark evaluation
model_name = "your_model_name"
json_path = f"path/to/your/benchmark/result_{model_name}-think.json"
with open(json_path, "r", encoding='utf-8') as f:
    data = json.load(f)
current_index = find_last_generate(data)  # find the sample index to start from


# Batch evaluation
for index in range(current_index, len(data)):
    item = data[index]
    # emotion ranking case
    if "predict_emotion_all" in item:
        emotions = item["predict_emotion_list"].replace(" ", "").split(",")
        emotions = [emo.lower() for emo in emotions]
        thinking = _extract_answer_content(item["predict_emotion_all"])
        task = "Emotion ranking"
        documents = anchors[task]
        final_state, score_dict = embedding_judge(prompt_template[3], thinking, documents, emotions, task)
        reward_score = similarity_score(final_state, emotions, 3) * (len(emotions) / 3)
        item["think_emotion"] = str(final_state)
        item["think_emotion_score"] = reward_score
        item["think_emotion_score_dict"] = score_dict
    if "predict_valence_all" in item:
        num_with = item["valence_predict"]
        thinking = _extract_answer_content(item["predict_valence_all"])
        task = "Valence"
        documents = anchors[task]
        final_state, score_dict = embedding_judge(prompt_template[0], thinking, documents, num_with, task)
        reward_score = similarity_score(final_state, num_with, 0)
        item["think_valence"] = str(final_state)
        item["think_valence_score"] = reward_score
        item["think_valence_score_dict"] = score_dict
    if "predict_arousal_all" in item:
        num_with = item["arousal_predict"]
        thinking = _extract_answer_content(item["predict_arousal_all"])
        task = "Arousal"
        documents = anchors[task]
        final_state, score_dict = embedding_judge(prompt_template[1], thinking, documents, num_with, task)
        reward_score = similarity_score(final_state, num_with, 1)
        item["think_arousal"] = str(final_state)
        item["think_arousal_score"] = reward_score
        item["think_dominance_score_dict"] = score_dict
    if "predict_dominance_all" in item:
        num_with = item["dominance_predict"]
        thinking = _extract_answer_content(item["predict_dominance_all"])
        task = "Dominance"
        documents = anchors[task]
        final_state, score_dict = embedding_judge(prompt_template[2], thinking, documents, num_with, task)
        reward_score = similarity_score(final_state, num_with, 2)
        item["think_dominance"] = str(final_state)
        item["think_dominance_score"] = reward_score
        item["think_dominance_score_dict"] = score_dict
    if index % 10 == 0:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"{model_name} {index}/{len(data)} thinking embedding completed.")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
print(f"{model_name} {index}/{len(data)} thinking embedding completed.")

# Statistics
key_list = ["emotion", "valence", "arousal", "dominance"]
for key in key_list:
    if f'think_{key}_score' not in data[0]:
        continue
    scores = [item[f'think_{key}_score'] for item in data]
    average = sum(scores) / len(scores) if scores else 0
    print(f"{key} avg score: {average:.4f}")
