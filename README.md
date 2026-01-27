
# EEmo-Logic: A Unified Dataset and Multi-Stage Framework for Image-Evoked Emotion Assessment

<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:60%" src="Spotlight.png">
</div>

Discerning intensity nuances and multi-dimensional attributes in image-evoked emotions is pivotal for advancing model empathy and human-computer interaction. However, existing models remain confined to coarse-level perception with limited task diversity. 
To bridge this gap, we introduce **EEmoDB**, the largest image-<u>e</u>voked <u>emo</u>tion understanding instruction <u>d</u>ataset, featuring five analysis dimensions and five task categories for fine-grained interpretation. Specifically, we compile $1.2M$ question-answering (QA) pairs (EEmoDB-QA) from $125k$ images via automated generation, alongside a $36k$ instruction set (EEmoDB-Assess) curated from $25k$ images for fine-grained assessment. 
Furthermore, we propose **EEmo-Logic**, an **all-in-one** MLLM developed via instruction fine-tuning and task-customized Group Relative Preference Optimization (GRPO) with novel rewards. Extensive experiments demonstrate that **EEmo-Logic** achieves robust performance across in-domain and cross-domain benchmarks, excelling in QA and fine-grained attribute assessment. 