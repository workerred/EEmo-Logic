
<h1>
EEmo-Logic: A Unified Dataset and Multi-Stage Framework for Image-Evoked Emotion Assessment
</h1>

<div align="center">
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:60%" src="Spotlight.png">
</div>
</div>

Discerning intensity nuances and multi-dimensional attributes in image-evoked emotions is pivotal for advancing model empathy and human-computer interaction. However, existing models remain confined to coarse-level perception with limited task diversity. 

To bridge this gap, we introduce **EEmoDB**, the largest image-<u>e</u>voked <u>emo</u>tion understanding instruction <u>d</u>ataset, featuring five analysis dimensions and five task categories for fine-grained interpretation. Specifically, we compile $1.2M$ question-answering (QA) pairs (**EEmoDB-QA**) from $125k$ images via automated generation, alongside a $36k$ instruction set (**EEmoDB-Assess**) curated from $25k$ images for fine-grained assessment. 

Furthermore, we propose **EEmo-Logic**, an **all-in-one** MLLM developed via instruction fine-tuning and task-customized Group Relative Preference Optimization (GRPO) with novel rewards. Extensive experiments demonstrate that **EEmo-Logic** achieves robust performance across in-domain and cross-domain benchmarks, excelling in QA and fine-grained attribute assessment. 

## 🔍 Qualitative Results

<div align="center">
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="Qualitative results.png">
</div>
</div>

## 📜TODO

- [ ] Release the training script
- [ ] Release the EEmo-Logic checkpoint
- [ ] Release the EEmoDB dataset
- [x] Release the inference script

## 🛠️ Installation

```shell
# create conda environment
conda create -n eemo-logic python=3.10
conda activate eemo-logic

# install requirements
pip install -r requirements.txt
```

## 🚀 Inference

### In-Domain Task

Assuming you have already downloaded the [EEmo-Bench](https://github.com/workerred/EEmo-Bench) dataset, you can use the following command to obtain EEmo-Logic's responses for the **Perception**, **Ranking**, **Description**, and **Assessment** tasks.

<details>
  <summary>Inference arguments</summary>

- `model_dir` (str): Path to your downloaded **EEmo-Logic** checkpoint. 
- `json_file` (str): Path to the question-answer pair JSON files in your downloaded EEmo-Bench dataset. 
- `output_file` (str): The save path for the result JSON files.
- `image_folder` (str): Path to the images folder in your downloaded EEmo-Bench dataset. 
</details>

```shell
python inference/evaluate_perception_single.py  # Single-image emotion perception task
python inference/evaluate_perception_pair.py  # Paired-images emotion perception task
python inference/evaluate_description.py  # Emotion description task
python inference/evaluate_ranking.py  # Emotion ranking task with reasoning process
python inference/evaluate_vad.py  # VAD assessment task with reasoning process
```
### Cross-Domain Dataset
1. Dominant Emotion Classification
Once you have prepared the [Artphoto](https://www.imageemotion.org/) and [ArtEmis](https://github.com/optas/artemis) cross-domain datasets, you can use the following code to perform inference:

      <details>
      <summary>Inference arguments</summary>

      - `model_dir` (str): Path to your downloaded **EEmo-Logic** checkpoint. 
      - `root_dir` (str): Path to your dataset folder of all the downloaded files.
      - `input_json` (str): Path to the JSON file recording the dominant emotion labels for each test sample.
      - `output_json` (str): The save path for the result JSON files.
      </details>

      ```shell
      python inference/evaluate_Artphoto_think.py  # Generate Artphoto results with reasoning process
      python inference/evaluate_ArtEmis_think.py  # Generate ArtEmis results with reasoning process
      ```

2. Aesthetic Empathy Question-Answering Tasks
Once you have prepared the [AesBench AesE](https://github.com/yipoh/AesBench) and [UNIAA Sent.](https://github.com/KlingTeam/Uniaa) cross-domain benchmarks, you can use the following code to perform inference:

      <details>
      <summary>Inference arguments</summary>

      - `model_dir` (str): Path to your downloaded **EEmo-Logic** checkpoint. 
      - `image_folder` (str): Path to the images folder in your downloaded Aesthetic benchmarks' Empathy subset. 
      - `input_json` (str): Path to the question-answer pair JSON files in your downloaded Aesthetic benchmarks' Empathy subset. 
      - `output_json` (str): The save path for the result JSON files.
      </details>
      
      ```shell
      python inference/evaluate_AesBench_AesE.py  # AesBench AesE
      python inference/evaluate_ArtEmis_think.py  # UNIAA Sent.
      ```
