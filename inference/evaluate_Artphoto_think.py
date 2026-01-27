from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from modelscope import snapshot_download
import json
import os
import re


def qwen_evaluate(img_path, prompt):
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{img_path}",
                },
                {"type": "text", "text": prompt}
            ],
        }
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(message)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(f"(Answer) {output_text[0]}")
    return output_text[0]


def build_emotion_dataset_json(root_dir, save_path="data.json"):
    dataset = []
    id = 0
    image_names = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
    for image in image_names:
        emotion = image.split("_")[0]
        image_path = os.path.join(root_dir, image)
        if os.path.isfile(image_path):
            data_item = {
                "id": id,
                "emotion": emotion,
                "image_path": image_path.replace("\\", "/")
            }
            dataset.append(data_item)
            id += 1

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"✅ {len(dataset)} samples have been saved to {save_path}")
    return dataset


def load_data(output_path):
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)
    else:
        output_data = build_emotion_dataset_json(root_dir=root_dir, save_path=output_path)
    return output_data


def find_last_generate(output_data, model_name):
    for i, item in enumerate(output_data):
        if f"predict({model_name})" not in item:
            return i
    return len(output_data)


def extract_answer_content(text):
    pattern = r'<answer>Dominant emotion:(.*?)</answer>(?!.*<answer>)'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


# ----------------------------------------------------------------------------------------
model_dir = "path/to/EEmo-Logic"
model_name = f"EEmo-Logic"
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="auto", device_map="auto"
)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels)
root_dir = "path/to/Artphoto"
dataset = "Artphoto"
output_json = f"path/to/Artphoto/{dataset}({model_name}).json"
output_data = load_data(output_json)
current_index = find_last_generate(output_data, model_name)
emotion_options = list(set([item["emotion"] for item in output_data]))
emotion_options_str = "[" + ", ".join(emotion_options) + "]"
QUESTION_TEMPLATE = """Answer the question: "[QUESTION]" according to the content of the image. Firstly, give a through thinking on this and then output your thought and analysis process within the <think> </think> tags and provide your answer within the <answer> </answer> tags (formatted as Dominant emotion:  ). The final output should be with the following format:
    <think></think><answer>Dominant emotion:  </answer>"""
for index in range(current_index, len(output_data)):
    item = output_data[index]
    img_name = item["image_path"]
    img_path = os.path.join(root_dir, img_name)
    prompt = "Assume you are an expert in emotional psychology. Based on the given image, carefully identify " \
             "the dominant emotion evoked by the image. " \
             f"The emotions to consider include: {emotion_options_str}" \
             "Please show your predicted dominant emotion directly as follows: "
    prompt = QUESTION_TEMPLATE.replace("[QUESTION]", prompt)
    answer = qwen_evaluate(img_path, prompt)
    item[f"all({model_name})"] = answer
    item[f"predict({model_name})"] = extract_answer_content(answer).replace(" ", "").lower()
    if index % 10 == 0:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
    print("{} {}/{} has completed the dominant emotion classification.".format(model_name, index, len(output_data)))
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)
print("{} {}/{} has completed the dominant emotion classification.".format(model_name, index, len(output_data)))
print(f"The evaluation of {model_name} has been completed.")
