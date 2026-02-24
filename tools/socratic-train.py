#!/usr/bin/env python3
"""
Socratic Training Sidecar for llama.cpp

Trains a QLoRA adapter from Socratic dialogue turns using PyTorch + PEFT,
then converts the result to GGUF format for hot-loading into llama.cpp.

Usage:
    python3 tools/socratic-train.py \
        --data /tmp/socratic_train_42.json \
        --model-path /path/to/base/model.gguf \
        --output adapters/socratic_20260224 \
        --result /tmp/socratic_train_42.json.result.json

The --data JSON has the schema:
{
    "turns": [{"prompt": "...", "response": "...", "signals": {...}}, ...],
    "output_path": "adapters/socratic_20260224",
    "config": {
        "learning_rate": 1e-4,
        "iterations": 100,
        "lora_rank": 16,
        "lora_alpha": 32
    }
}

The --result JSON written on completion:
{
    "success": true/false,
    "adapter_path": "...",
    "before_perplexity": float,
    "after_perplexity": float,
    "improvement": float,
    "turns_used": int,
    "training_time_seconds": float,
    "error": "..." (only on failure)
}

Dependencies (expected on GH200): torch, transformers, peft, bitsandbytes, datasets
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time


def compute_perplexity(model, tokenizer, texts, max_length=512):
    """Compute perplexity on a list of texts."""
    import torch

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(model.device)

            outputs = model(**encodings, labels=encodings["input_ids"])
            n_tokens = encodings["input_ids"].numel()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def resolve_hf_model(gguf_path):
    """
    Resolve a GGUF model path to its HuggingFace model ID.
    For quantized GGUF models, we need the original HF model for training.
    """
    # Common mappings for models we use
    name_lower = os.path.basename(gguf_path).lower()

    if "llama-3.3-70b" in name_lower or "llama-3-3-70b" in name_lower:
        return "meta-llama/Llama-3.3-70B-Instruct"
    if "llama-3.1-70b" in name_lower or "llama-3-1-70b" in name_lower:
        return "meta-llama/Meta-Llama-3.1-70B-Instruct"
    if "llama-3.1-8b" in name_lower or "llama-3-1-8b" in name_lower:
        return "meta-llama/Meta-Llama-3.1-8B-Instruct"
    if "qwen" in name_lower and "72b" in name_lower:
        return "Qwen/Qwen2.5-72B-Instruct"
    if "qwen" in name_lower and "32b" in name_lower:
        return "Qwen/Qwen2.5-32B-Instruct"
    if "mistral" in name_lower:
        return "mistralai/Mistral-7B-Instruct-v0.3"

    # Fallback: try the path as-is (might be an HF path already)
    return gguf_path


def main():
    parser = argparse.ArgumentParser(description="Socratic Training Sidecar")
    parser.add_argument("--data", required=True, help="Path to training data JSON")
    parser.add_argument("--model-path", required=True, help="Path to base model (GGUF or HF)")
    parser.add_argument("--output", required=True, help="Output adapter directory")
    parser.add_argument("--result", required=True, help="Path to write result JSON")
    args = parser.parse_args()

    result = {
        "success": False,
        "adapter_path": args.output,
        "before_perplexity": 0.0,
        "after_perplexity": 0.0,
        "improvement": 0.0,
        "turns_used": 0,
        "training_time_seconds": 0.0,
        "error": "",
    }

    start_time = time.time()

    try:
        # Load training data
        with open(args.data) as f:
            data = json.load(f)

        turns = data.get("turns", [])
        config = data.get("config", {})
        result["turns_used"] = len(turns)

        if not turns:
            result["error"] = "No training turns provided"
            _write_result(args.result, result)
            return 1

        # Training hyperparameters
        learning_rate = config.get("learning_rate", 1e-4)
        iterations = config.get("iterations", 100)
        lora_rank = config.get("lora_rank", 16)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.05)
        batch_size = config.get("batch_size", 1)
        max_seq_length = config.get("max_seq_length", 2048)

        # Resolve HuggingFace model ID from GGUF path
        hf_model_id = resolve_hf_model(args.model_path)
        print(f"[socratic-train] Base model: {hf_model_id}")
        print(f"[socratic-train] Training {len(turns)} turns, lr={learning_rate}, rank={lora_rank}")

        # Import heavy dependencies only when needed
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import Dataset

        # Load model with 4-bit quantization (QLoRA)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        print("[socratic-train] Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        # Prepare training texts
        texts = []
        for turn in turns:
            prompt = turn.get("prompt", "")
            response = turn.get("response", "")
            weight = turn.get("signals", {}).get("consolidation_weight", 1.0)
            # Format as instruction-response pair
            text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
            texts.append(text)

        # Compute baseline perplexity
        eval_texts = texts[:min(5, len(texts))]
        result["before_perplexity"] = compute_perplexity(model, tokenizer, eval_texts, max_seq_length)
        print(f"[socratic-train] Before perplexity: {result['before_perplexity']:.4f}")

        # Prepare for QLoRA training
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # Create dataset
        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
            )

        dataset = Dataset.from_dict({"text": texts})
        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

        # Training
        from transformers import TrainingArguments, Trainer

        hf_output_dir = os.path.join(args.output, "hf_adapter")
        os.makedirs(hf_output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=hf_output_dir,
            num_train_epochs=max(1, iterations // max(1, len(texts))),
            max_steps=iterations,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=config.get("weight_decay", 0.01),
            logging_steps=10,
            save_strategy="no",
            bf16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=lambda data: {
                "input_ids": torch.stack([torch.tensor(d["input_ids"]) for d in data]),
                "attention_mask": torch.stack([torch.tensor(d["attention_mask"]) for d in data]),
                "labels": torch.stack([torch.tensor(d["input_ids"]) for d in data]),
            },
        )

        print(f"[socratic-train] Starting training ({iterations} steps)...")
        trainer.train()

        # Save HuggingFace adapter
        model.save_pretrained(hf_output_dir)
        tokenizer.save_pretrained(hf_output_dir)
        print(f"[socratic-train] Saved HF adapter to {hf_output_dir}")

        # Compute post-training perplexity
        result["after_perplexity"] = compute_perplexity(model, tokenizer, eval_texts, max_seq_length)
        print(f"[socratic-train] After perplexity: {result['after_perplexity']:.4f}")

        if result["before_perplexity"] > 0:
            result["improvement"] = (result["before_perplexity"] - result["after_perplexity"]) / result["before_perplexity"]
        print(f"[socratic-train] Improvement: {result['improvement']:.4f}")

        # Convert to GGUF
        gguf_output = os.path.join(args.output, "adapter.gguf")
        os.makedirs(args.output, exist_ok=True)

        # Use llama.cpp's convert_lora_to_gguf.py
        llama_cpp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        convert_script = os.path.join(llama_cpp_dir, "convert_lora_to_gguf.py")

        if os.path.exists(convert_script):
            print(f"[socratic-train] Converting to GGUF: {gguf_output}")
            convert_cmd = [
                sys.executable, convert_script,
                "--outfile", gguf_output,
                "--base", hf_model_id,
                hf_output_dir,
            ]
            subprocess.run(convert_cmd, check=True)
            result["adapter_path"] = gguf_output
        else:
            # Fallback: keep HF format
            print(f"[socratic-train] Warning: convert_lora_to_gguf.py not found, keeping HF format")
            result["adapter_path"] = hf_output_dir

        result["success"] = True
        result["training_time_seconds"] = time.time() - start_time
        print(f"[socratic-train] Training complete in {result['training_time_seconds']:.1f}s")

    except Exception as e:
        result["error"] = str(e)
        result["training_time_seconds"] = time.time() - start_time
        print(f"[socratic-train] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

    _write_result(args.result, result)
    return 0 if result["success"] else 1


def _write_result(path, result):
    """Write result JSON to disk."""
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
