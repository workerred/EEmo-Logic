# EEmo-Bench: AI Agent Instructions

## Project Overview

**EEmo-Bench** is a benchmark for evaluating Multi-modal Large Language Models (MLLMs) on image-evoked emotion assessment. The codebase focuses on inferencing and evaluation scripts that run MLLMs against structured datasets to measure their emotion understanding capabilities across four task types:
- **Perception**: Emotion/VAD attribute recognition from single or paired images
- **Ranking**: Ranking emotions by intensity (top 3 of 7 basic emotions)
- **Description**: Descriptive and comparative analysis of image emotions
- **Assessment**: Quantifying VAD scores (Valence, Arousal, Dominance)

## Architecture & Data Flow

### Core Evaluation Pipeline
All evaluation scripts (`inference/evaluate_*.py`) follow this pattern:
1. **Model Loading**: Load Qwen2.5-VL from HuggingFace with specific configs (flash_attention_2, bfloat16, auto device mapping)
2. **Prompt Composition**: Build task-specific prompts with domain context ("expert in emotional psychology")
3. **Inference**: Process images through model, extract structured responses using regex
4. **Results Persistence**: Save predictions to JSON, resume from last completed index on restart

### Data Structures
- **Input JSON**: `{question, option/emotion_list, image_name, image_path, ...}`
- **Output JSON**: Appends predictions to existing file with fields like `predict_emotion_list`, `valence_predict`, etc.
- **Resume Mechanism**: Scripts check for output file existence and find last non-predicted entry

## Key Patterns & Conventions

### Model Initialization Pattern
```python
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_dir, min_pixels=256*28*28, max_pixels=1280*28*28)
```
- Always use `torch.bfloat16` for memory efficiency
- Always use `device_map="auto"` for multi-device scaling
- Pixel ranges tuned for VAD assessment (token budget: 256-1280 per image)

### Prompt Engineering Pattern
- **Prefix**: "Assume you are an expert in emotional psychology..."
- **Emotion List**: 7 basic emotions (anger, disgust, fear, joy, neutral, sadness, surprise)
- **Response Wrapper**: Model outputs wrapped in `<answer>...</answer>` tags (extracted via regex: `r'<answer>(.*?)</answer>(?!.*<answer>)'`)
- **Multi-image Support**: Single and paired images handled separately (different prompt templates)

### Dataset Paths Convention
- Input: `path/to/EEmo-Bench(single).json` or `EEmo-Bench(pair).json`
- Output: `path/to/EEmo-Bench({model_name}-single).json` or `-pair.json`
- Model directory: `path/to/EEmo-Logic` (usually points to local HF model dir)

## Critical Configuration Points

### GPU Memory Optimization
- **Flash Attention 2**: Reduces memory footprint; disable if unsupported
- **Pixel Budget**: 256-1280 controls token count per image; increase for better quality, decrease for speed
- **max_new_tokens**: Set to 1024 for all tasks (sufficient for rankings/descriptions)

### Resume & Checkpoint Logic
- **Single-image tasks**: Check for `"image_name"` in output (numeric filename)
- **Ranking task**: Check for `"predict_emotion_list"` key
- **VAD task**: Check for `"valence_predict"` key
- Always append to existing output file to preserve prior work

## Common Modifications for New Models

When adapting scripts for different MLLMs:
1. Replace model/processor loading (keep transformer import pattern)
2. Adapt `compose_prompt()` function if model requires different instruction formats
3. Adjust `max_new_tokens` based on model's capabilities
4. Update `extract_answer_content()` regex if model uses different response markers
5. Test on single image first, then scale to dataset

## Testing & Validation

### Expected Outputs
- All evaluation scripts produce JSON with model predictions aligned to question indices
- VAD predictions: correlation metrics (Pearson, Spearman) against ground truth
- Ranking predictions: top-3 emotion accuracy vs manual annotations
- Description predictions: evaluated via GPT/LLM-as-judge (see README for pseudo code)

### Debug Flags
- Uncomment `# os.environ['CUDA_VISIBLE_DEVICES'] = '3'` to pin GPU
- Print statements capture: `[Prompt]`, `(Answer)` for tracing inference flow
- JSON comparison: existing vs new output files to catch schema changes

## Integration Points

- **External**: HuggingFace model hub, ModelScope snapshot_download
- **Cross-task**: Shared `qwen_evaluate()` function signature (img_path, prompt → output_text)
- **Evaluation**: Results feed into leaderboard system (see README leaderboards/)
