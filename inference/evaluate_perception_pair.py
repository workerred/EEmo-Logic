from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from modelscope import snapshot_download
import json
import os


def qwen_evaluate(img_path1, img_path2, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{img_path1}"},
                {"type": "image", "image": f"file://{img_path2}"},
                {"type": "text", "text": f"{prompt}"},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
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
        output_data = []
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)
    return input_data, output_data


def find_last_generate(output_data):
    if not output_data:
        current_index = 0
    else:
        current_index = len(output_data)
    return current_index


def compose_prompt(question, option):
    assume_str = "Assume you are an expert in emotional psychology. Based on the two given images, please analyze the following emotional " \
                 "psychology questions. Be sure to understand the emotional attributes each image evokes in the viewer " \
                 "and the emotional relationship between these two images in your answers. " \
                 "The emotions to be considered include anger, disgust, fear, joy, neutral, sadness, and surprise. " \
                 "The questions are as follows:"
    option_list = [key+":"+option[key] for key in list(option.keys())]
    option_str = ' '.join(option_list)
    tips = " Choose between one of the following options: "
    prompt = assume_str + question + tips + option_str
    print(f"[{prompt}]")
    return prompt


# -----------------------main---------------------------
model_dir = "path/to/EEmo-Logic"

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels)

model_name = "EEmo-Logic"
json_file = "path/to/EEmo-Bench(pair).json"
output_file = f"path/to/EEmo-Bench({model_name}-pair).json"

input_data, output_data = load_data(json_file, output_file)
current_index = find_last_generate(output_data)
image_folder = "path/to/EEmo-bench/images"

for index in range(current_index, len(input_data)):
    output_item = input_data[index]
    img_name_1 = output_item["image1_url"]
    img_name_2 = output_item["image2_url"]
    img_path_1 = os.path.join(image_folder, img_name_1)
    img_path_2 = os.path.join(image_folder, img_name_2)
    # judge
    j_question = output_item["judge_question"]
    j_option = output_item["judge_options"]
    j_prompt = compose_prompt(j_question, j_option)
    j_answer = qwen_evaluate(img_path_1, img_path_2, j_prompt)
    output_item[f"{model_name}(judge)"] = j_answer
    # choose
    c_question = output_item["choose_question"]
    c_option = output_item["choose_options"]
    c_prompt = compose_prompt(c_question, c_option)
    c_answer = qwen_evaluate(img_path_1, img_path_2, c_prompt)
    output_item[f"{model_name}(choose)"] = c_answer

    output_data.append(output_item)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print("Image pair {} is done.".format(index+1))
print("The evaluation has been completed.")
