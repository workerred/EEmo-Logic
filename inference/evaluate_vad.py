import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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


def load_data(input_path, output_path):
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)
    return output_data


def find_last_generate(data):
    for i in range(len(data)):
        if "valence_predict" not in data[i]:
            return i
    return len(data)


def extract_answer_content(text):
    pattern = r'<answer>(.*?)</answer>(?!.*<answer>)'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


# -----------------------main---------------------------
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

input_path = "path/to/EEmo-Bench(single).json"
Ebench_path = f"path/to/EEmo-Bench({model_name}-single).json"

output_data = load_data(input_path, Ebench_path)
current_index = find_last_generate(output_data)
image_folder = "path/to/EEmo-bench/images"

for index in range(current_index, len(output_data)):
    output_item = output_data[index]
    img_name = output_item["image_name"]
    img_path = os.path.join(image_folder, img_name)
    task_list = ["valence", "arousal", "dominance"]
    replace_content = ["Valence: x.xxxx", "Arousal: x.xxxx", "Dominance: x.xxxx"]
    QUESTION_TEMPLATE = """Answer the question: "[QUESTION]" according to the content of the image. Firstly, give a through thinking on this and then output your thought and analysis process within the <think> </think> tags and provide your answer within the <answer> </answer> tags (formatted as Score: ). The final output should be with the following format:
            <think></think><answer>Score: </answer>"""
    for i in range(len(task_list)):
        prompt = f"As an expert in emotional attribute assessment, please rate the level of {task_list[i]} the image evokes in you on a scale from 0 to 1. "
        prompt = QUESTION_TEMPLATE.replace("[QUESTION]", prompt)
        prompt = prompt.replace("Score: ", replace_content[i])
        answer = qwen_evaluate(img_path, prompt)
        output_item[f"predict_{task_list[i]}_all"] = answer
        try:
            extract_answer = float(extract_answer_content(answer).split(":")[-1][1:])
        except:
            extract_answer = extract_answer_content(answer)
            print(f"Match failed, the result is {extract_answer}")
        output_item[f"{task_list[i]}_predict"] = extract_answer

    output_data[index] = output_item
    with open(Ebench_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print("{} {} is done.".format(model_name, output_item["image_name"]))
print(f"{model_name} VAD prediction has been completed.")
