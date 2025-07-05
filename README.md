# Gemma-7B ChatGPT Prompts Fine-Tuning with Unsloth

This project demonstrates how to fine-tune Google's Gemma-7B language model using the Unsloth library for efficient LoRA (Low-Rank Adaptation) training. The workflow is implemented in the notebook `Unsloth_finetuning_gemma_3_4b_pt_lora.ipynb` and uses the [fka/awesome-chatgpt-prompts](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts) dataset to teach the model to respond to a variety of instructions and prompts in a ChatGPT-like format.

---

## Project Structure

```
.
├── Unsloth_finetuning_gemma_3_4b_pt_lora.ipynb   # Main notebook for fine-tuning
└── gemma-7b-chatgpt-prompts/                    # Output directory for model and tokenizer
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    ├── tokenizer.model
    └── README.md  # Model card and details
```

---

## Quickstart

1. **Install dependencies:**
   ```python
   !pip install unsloth trl peft accelerate bitsandbytes transformers datasets fsspec
   ```
2. **Run the notebook:**
   Open and execute `Unsloth_finetuning_gemma_3_4b_pt_lora.ipynb` step by step in a GPU-enabled environment (e.g., Google Colab).
3. **Export the model:**
   The fine-tuned model and tokenizer will be saved in the `gemma-7b-chatgpt-prompts/` directory.

---

## Model Details

- **Base Model:** `google/gemma-7b` (4-bit quantized via Unsloth)
- **LoRA Config:** r=64, alpha=128, dropout=0, target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Dataset:** [fka/awesome-chatgpt-prompts](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts)
- **Training:** 3 epochs, batch size 2, gradient accumulation 4, AdamW 8-bit optimizer

---

## Inference Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path = "gemma-7b-chatgpt-prompts"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

inputs = tokenizer("### Instruction: Give me a recipe for chocolate cake.\n### Prompt: \n### Response:", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```


- For questions about Unsloth, see the [Unsloth GitHub](https://github.com/unslothai/unsloth).

---

## License

Please check the licenses for Google Gemma, Unsloth, and the dataset before commercial use. 
