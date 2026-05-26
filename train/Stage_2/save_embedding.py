import json
import re
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch import Tensor


# ==========================================
# 0. Environment & Model Setup (update paths as needed)
# ==========================================
# Please replace with your actual Qwen2.5/Qwen3-Embedding model path
try:
    scale = "8B"
    model_path = f"path/to/Qwen3-Embedding-{scale}"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda")
    model.eval()
except:
    print("WARNING: Model not loaded. Embedding computation will fail. Please configure MODEL_PATH.")
    tokenizer = None
    model = None


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


# ==========================================
# Core function: Embedding computation
# ==========================================
def embedding_judge(task, description, documents):
    """
    Compute similarity between description (Query) and documents (Keys).
    """
    max_length = 8192

    # Construct input: [Query, Doc1, Doc2, ...]
    queries = [
        get_detailed_instruct(task, description)
    ]
    input_texts = queries + documents

    # Tokenize
    batch_dict = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}

    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Compute similarity: Query (index 0) vs Documents (index 1 to end)
    scores = (embeddings[:1] @ embeddings[1:].T).squeeze(0).tolist()

    return scores


# Min-Max normalization
def min_max_normalize_matrix(matrix, power=1.0):
    """
    Row-wise Min-Max normalization of a similarity matrix.
    Formula: (x - min) / (max - min)

    Args:
        matrix: List[List[float]] or np.ndarray, shape [N, N]
        power: float, sharpening exponent.
               1.0 = linear normalization (preserve original ratios)
               >1.0 (e.g., 3.0) = suppress mid-low scores, highlight high scores
    Returns:
        np.ndarray: Normalized matrix
    """
    mat = np.array(matrix)

    # 1. Get the max (usually self) and min (least related emotion) of each row
    row_maxs = mat.max(axis=1, keepdims=True)
    row_mins = mat.min(axis=1, keepdims=True)

    # 2. Compute denominator (Range)
    # Defensive: prevent division by zero
    denominators = row_maxs - row_mins
    denominators[denominators == 0] = 1.0

    # 3. Apply Min-Max normalization
    normalized_mat = (mat - row_mins) / denominators

    # 4. (Optional) Exponential sharpening
    # If you want 0.5 to become 0.125 (0.5^3), set power=3
    if power != 1.0:
        normalized_mat = np.power(normalized_mat, power)
    normalized_mat = normalized_mat.tolist()

    return normalized_mat


# ==========================================
# 1. Emotion definition dictionary
# ==========================================
emotion_definitions = {
    # ==================================================
    # 1. Joy & High Arousal
    # ==================================================
    'joy': "A positive, moderate energy state with a sense of control. Response to immediate success or delightful stimuli. Smiling, laughter, and sudden uplift in energy. Engagement with the moment.",
    'happiness': "A positive, high energy state of feeling dominant and secure. Sustained evaluation that life conditions are favorable. General sense of stability, satisfaction, and well-being.",
    'excitement': "A positive, high energy state with a feeling of empowerment. Anticipation of a stimulating future event. High physical energy, eagerness, and focused attention on upcoming rewards.",
    'ecstasy': "A positive, high energy state with balanced control. Overwhelming peak experience where self-awareness dissolves. Trance-like state induced by extreme sensory intensity.",
    'amusement': "A positive, moderate energy state with a sense of safety. Response to humor or incongruity. Lighthearted laughter and playfulness without serious stakes.",
    'pleasure': "A positive, high energy state of gratification and control. Sensory gratification from physical stimuli (taste, touch) or aesthetic beauty. Immediate comfort and enjoyment.",
    'amazement': "A positive, high energy state characterized by being overwhelmed yet impressed. Response to something unexpected. Momentary suspension of action. Wide eyes and jaw drop.",
    'awe': "A positive, low energy state of feeling small but safe. Reaction to vastness or grandeur that transcends current understanding. Goosebumps and quiet reverence.",

    # ==================================================
    # 2. Peace & Contentment
    # ==================================================
    'peace': "A positive, moderate energy state with a strong sense of security. Absence of conflict, disturbance, or war. Social harmony and freedom from external commotion.",
    'serenity': "A positive, low energy state with balanced composure. Deep inner stillness and clarity of mind. Unruffled composure regardless of external chaos.",
    'content': "A positive, low energy state of simple acceptance. Perception that current resources are sufficient for needs. Absence of desire for more.",
    'contentment': "A positive, moderate energy state with a sense of stability. Long-term acceptance of one's lot in life. Gratitude for stability and lack of friction.",
    'acceptance': "A positive, moderate energy state of acknowledging reality. Cognitive consent to reality without resistance. Acknowledging facts without attempting to change them.",
    'relief': "A positive, moderate energy state of release. Reaction to the removal of a threat, pressure, or burden. Exhaling, muscle relaxation, and cessation of vigilance.",

    # ==================================================
    # 3. Sadness & Suffering
    # ==================================================
    'sad': "A negative, low energy state feeling powerless. Response to a specific unpleasant event or minor disappointment. Temporary frowning and lowering of spirits.",
    'sadness': "A negative, low energy state of withdrawal and helplessness. Response to irrevocable loss or failure. Heaviness, tears, lethargy, and withdrawal from social interaction.",
    'grief': "A negative, moderate energy state of deep distress and vulnerability. Reaction to profound bereavement or death. Deep, sharp distress, sobbing, and the process of mourning.",
    'suffering': "A negative, moderate energy state of enduring hardship without control. Endurance of prolonged pain or adversity. Feeling of being subjected to torment over time.",
    'pain': "A negative, high energy state of acute distress and vulnerability. Sensation of physical hurt or deep emotional wounding. Signal of damage requiring attention or healing.",
    'yearning': "A neutral, low energy state of longing. Intense desire for something or someone that is absent. Focus on the gap between reality and the desired object.",
    'fatigue': "A negative, low energy state of depletion. Depletion of physical or mental resources. Inability to continue effort due to tiredness. Heavy limbs.",
    'boredom': "A negative, low energy state with stagnant focus. Perception of the environment as static and lacking stimulation. Restless desire for engagement but inability to find it.",

    # ==================================================
    # 4. Fear & Anxiety
    # ==================================================
    'fear': "A negative, high energy state of feeling threatened and powerless. Response to an immediate, concrete threat. Activation of fight-or-flight survival mechanism. Urge to escape.",
    'terror': "A negative, high energy state of total overwhelm and helplessness. Overwhelming reaction to imminent mortal danger. Paralysis, screaming, or collapse of rational thought.",
    'apprehension': "A neutral, moderate energy state of uncertainty. Uneasy anticipation of a potential future negative event. Worry about uncertain outcomes.",
    'disquietment': "A negative, high energy state of restlessness. General restlessness and lack of calm. A subtle, nagging sense that something is wrong.",
    'vigilance': "A neutral, low energy state of watchful control. Sustained attention to detect potential signals or threats. Guarded behavior and scanning the environment.",
    'panic': "A negative, high energy state of loss of control. Sudden, uncontrollable surge of hysteria. Hyperventilation, racing heart, and confusion driven by perceived entrapment.",

    # ==================================================
    # 5. Anger & Disgust
    # ==================================================
    'anger': "A negative, moderate energy state with an urge to confront. Response to perceived injustice, offense, or goal obstruction. Urge to punish or remove the obstacle.",
    'rage': "A negative, high energy state of explosive volatility. Violent reaction to provocation. Loss of impulse control, screaming, and destructive urges.",
    'annoyance': "A negative, moderate energy state of irritation. Reaction to a mild nuisance or repeated disturbance. Low-level irritation and desire for the noise to stop.",
    'disapproval': "A negative, moderate energy state of judgment. Judgmental rejection of an action or idea based on moral or standard violation. Shaking head.",
    'disgust': "A negative, moderate energy state of rejection. Visceral rejection of something toxic, contaminated, or morally offensive. Nausea and urge to turn away.",
    'loathing': "A negative, moderate energy state of deep revulsion. Deep-seated, intense hatred. Total rejection of the existence of the subject.",
    'aversion': "A neutral, moderate energy state of avoidance. Strong desire to avoid or turn away from a specific stimulus. Active disinclination to interact.",

    # ==================================================
    # 6. Social Emotions
    # ==================================================
    'affection': "A positive, moderate energy state of warmth and security. Warm attachment towards another person. Physical closeness, hugging, tenderness, and caregiving.",
    'admiration': "A positive, moderate energy state of looking up to someone. Recognition of superior qualities in another. Respect and wonder.",
    'esteem': "A positive, low energy state of deep respect. Valuing the worth or dignity of a person (self or other). High regard.",
    'trust': "A positive, moderate energy state of reliance and safety. Reliance on the integrity or strength of someone. Willingness to be vulnerable.",
    'confidence': "A positive, moderate energy state of self-assurance. Self-assurance in one's own abilities or judgment. Certainty in action and posture.",
    'sympathy': "A positive, low energy state of resonance. Resonance with another person's misfortune. Feeling pity or sorrow for someone else's situation.",
    'embarrassment': "A negative, moderate energy state of self-consciousness and vulnerability. Reaction to a social blunder. Blushing and desire to hide.",

    # ==================================================
    # 7. Cognitive & Complex
    # ==================================================
    'interest': "A positive, moderate energy state of focused engagement. Focus of attention on a novel stimulus. Desire to explore, learn, or understand more.",
    'engagement': "A positive, moderate energy state of flow. State of being fully occupied and absorbed in an activity. Active participation.",
    'anticipation': "A neutral, moderate energy state of expectancy. Looking forward to a future event. Mental preparation and expectation of an outcome.",
    'surprise': "A positive, high energy state of sudden reaction. Reaction to an unexpected event. Momentary interruption of thought processing. Startle response.",
    'doubt/confusion': "A negative, moderate energy state of uncertainty and hesitation. Lack of certainty or clarity. Inability to interpret signals.",
    'pensiveness': "A negative, moderate energy state of internal reflection. State of deep, serious reflection. Withdrawal of attention from the outside world.",
    'distraction': "A neutral, low energy state of fragmented focus. Fragmented attention. Inability to maintain focus on the primary task due to interference.",
    'disconnection': "A negative, low energy state of detachment. Sense of detachment from reality. Numbness, dissociation, or feeling separated by a wall.",
    'sensitivity': "A neutral, moderate energy state of heightened responsiveness. Quick detection of subtle changes in external stimuli or emotional cues.",
    'neutral': "A neutral, low energy state of equilibrium. Baseline state. Absence of strong activation or valence. Calmness.",
}

# ==========================================
# 2. Read data and extract unique options
# ==========================================
input_file = "path/to/your/RL_train_data.json"
output_file = "check_embedding.json"

# Prepend emotion name: Joy: ...
for key in emotion_definitions.keys():
    emotion_definitions[key] = str(key).capitalize() + ": " + emotion_definitions[key]

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
except FileNotFoundError:
    print(f"Error: {input_file} not found.")
    raw_data = []

# Store deduplicated option strings
unique_options = set()

# Regex to match content inside parentheses: joy(contentment, sadness) -> contentment, sadness
pattern = re.compile(r'\((.*?)\)')

for item in raw_data:
    if 'answer' in item:
        match = pattern.search(item['answer'])
        if match:
            # Extract parentheses content, e.g., "disgust, contentment, sadness..."
            content = match.group(1).strip()
            if content:
                unique_options.add(content)

# Initialize check_dict
check_dict = {opt: [] for opt in unique_options}
print(f"Extracted {len(check_dict)} unique option combinations.")

# ==========================================
# 3. Iterate to compute correlation matrices
# ==========================================
task_instruction = (
    "Represent the emotional definition to identify the situational trigger "
    "(what caused it), cognitive appraisal (how it is interpreted), and "
    "behavioral response (action tendency)."
)

for options_str in check_dict:
    # 1. Split into individual emotion terms
    emotions_list = [e.strip() for e in options_str.split(',')]

    # 2. Get corresponding definition list
    definitions_list = []
    valid_indices = []  # track valid emotion indices to prevent key errors

    for idx, emo in enumerate(emotions_list):
        # Handle potential case/sensitivity and whitespace issues
        clean_emo = emo.lower()
        if clean_emo in emotion_definitions:
            definitions_list.append(emotion_definitions[clean_emo])
            valid_indices.append(idx)
        else:
            print(f"Warning: Definition for '{emo}' not found, skipping.")
            definitions_list.append(f"Definition of {emo}")  # Fallback

    # 3. Compute N x N correlation matrix
    # Row i represents: i-th emotion definition vs [all emotion definitions] similarity
    matrix = []

    if model is not None:
        print(f"Computing combination: {options_str}...")
        for i, target_def in enumerate(definitions_list):
            # Call embedding_judge function
            # query: current emotion definition
            # documents: all emotion definitions in the list
            row_scores = embedding_judge(task_instruction, target_def, definitions_list)
            matrix.append(row_scores)
    else:
        # Mock data provided if model is not loaded
        matrix = [[0.0] * len(definitions_list) for _ in range(len(definitions_list))]

    # 4. Save results
    # Normalize each emotion's scores
    check_dict[options_str] = min_max_normalize_matrix(matrix)

# ==========================================
# 4. Save and print
# ==========================================
# Write JSON
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(check_dict, f, indent=4, ensure_ascii=False)

print(f"\nComputation complete. Results saved to {output_file}")

# Print partial results preview
print("\n--- Results Preview ---")
for key, val in list(check_dict.items())[:2]:  # only print first two to avoid flooding
    print(f"Options: {key}")
    arr = np.array(val)
    print("Correlation Matrix:")
    print(arr)
    print("-" * 30)
