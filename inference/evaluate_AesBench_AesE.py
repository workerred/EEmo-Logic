import torch
from PIL import Image
device = "cuda" if torch.cuda.is_available() else "cpu"
import json, time, os
import PIL.Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


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
    return output_text[0]


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
folder_path = r"path/to/EAPD/images"
save_name = f"path/to/EAPD/test_AesE({model_name}).json"
bench_path = "path/to/EAPD/AesBench_AesE.json"
f = open(bench_path, encoding='utf-8')
data=json.load(f)
f.close()
answers={}
all_num = len(data)

img_num = 1
start_time = time.time()
#####-------AesE--------------------------
for imgName, label in data.items():
    print()
    print(imgName)
    img_path = os.path.join(folder_path, imgName)

    AesE_data = label['AesE_data']
    AesE_prompt = AesE_data['Question'] + " Choose one from the following options:\n" + AesE_data['Options'] + "\nYou should output a correct option without explanation."
    print(AesE_prompt)
    start = time.time()
    time.sleep(1)
    AesE_message = qwen_evaluate(img_path, AesE_prompt)
    print(AesE_message)
    answers[imgName] = {"AesE_response": AesE_message}
    avg_time = (time.time() - start_time) / img_num
    need_time = (avg_time * (all_num - img_num)) / 3600
    answers_dict = json.dumps(answers, indent=4)
    with open(save_name, 'w') as outfile:
        outfile.write(answers_dict)
    print(
        "AesE--{}/{} finished. Using time (s):{:.1f}. Average image time (s):{:.1f}. Need time (h):{:.1f}.".format(
            img_num, all_num, time.time() - start, avg_time, need_time))
    img_num = img_num + 1
