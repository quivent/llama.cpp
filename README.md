<div align="center">

```
 _ _                             ____  ____  
| | | __ _ _ __ ___   __ _      / ___||  _ \ 
| | |/ _` | '_ ` _ \ / _` |____| |    | |_) |
| | | (_| | | | | | | (_| |____| |___ |  __/ 
|_|_|\__,_|_| |_| |_|\__,_|     \____||_|    
```

**llama.cpp**

*Socratic KV signals fork of llama.cpp — 9 signal types + adapter lifecycle + training endpoints*

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?style=for-the-badge)](https://github.com/ggml-org/llama.cpp/releases)
[![C++](https://img.shields.io/badge/C++-11-blue.svg?style=for-the-badge&logo=c%2B%2B)](https://isocpp.org/)

LLM inference in C/C++
</div>

---

## 📑 Table of Contents

- [⚡ Overview](#-overview)
- [✨ Features](#-features)
- [📦 Quick Start](#-quick-start)
- [🚀 Tools](#-tools)
- [🔧 Hardware Support](#-hardware-support)
- [📖 Ecosystem](#-ecosystem)

---

## ⚡ Overview

The main goal of `llama.cpp` is to enable LLM inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud.

> [!NOTE]
> This fork implements Socratic KV signals — supporting 9 signal types, adapter lifecycles, and training endpoints.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- 1.5-bit through 8-bit integer quantization for faster inference and reduced memory use
- CPU+GPU hybrid inference to partially accelerate models larger than total VRAM capacity

---

## ✨ Features

<details>
<summary>Supported Text Models</summary>

- LLaMA 1/2/3 🦙
- Mistral 7B / Mixtral MoE
- DBRX, Jamba, Falcon, Qwen, Deepseek
- Gemma, Mamba, Grok-1, Command-R
- Phi models, GPT-2, InternLM2
- And many more...
</details>

<details>
<summary>Supported Multimodal Models</summary>

- LLaVA 1.5 / 1.6
- BakLLaVA, Obsidian, ShareGPT4V
- MobileVLM, Yi-VL, Mini CPM
- Moondream, Bunny, Qwen2-VL
</details>

<details>
<summary>Supported Bindings</summary>

Available in Python, Go, Node.js, JS/TS, Wasm, Ruby, Rust, C#, Scala, Clojure, React Native, Java, Zig, Flutter/Dart, PHP, Guile Scheme, Swift, Delphi, and Android.
</details>

---

## 📦 Quick Start

Getting started with llama.cpp is straightforward. Here are several ways to install it:

- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Run with Docker
- Install via `brew`, `nix` or `winget`

```sh
# Use a local model file
llama-cli -m my_model.gguf

# Or download and run a model directly from Hugging Face
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

# Launch OpenAI-compatible API server
llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

> [!TIP]
> `llama.cpp` requires the model to be stored in the GGUF file format. Models in other formats can be converted to GGUF using the `convert_*.py` Python scripts in this repo.

---

## 🚀 Tools

### `llama-cli`
A CLI tool for accessing and experimenting with most of `llama.cpp`'s functionality.
```bash
# Run in conversation mode
llama-cli -m model.gguf -cnv --chat-template chatml

# Constrain the output with a custom grammar
llama-cli -m model.gguf -n 256 --grammar-file grammars/json.gbnf -p 'Request: schedule a call at 8pm; Command:'
```

### `llama-server`
A lightweight, OpenAI API compatible, HTTP server for serving LLMs.
```bash
# Start a local HTTP server
llama-server -m model.gguf --port 8080

# Serve an embedding model
llama-server -m model.gguf --embedding --pooling cls -ub 8192
```

### `llama-perplexity` & `llama-bench`
Tools for measuring model perplexity and benchmarking inference performance.

---

## 🔧 Hardware Support

| Backend | Target devices |
| --- | --- |
| Metal | Apple Silicon |
| BLAS / BLIS | All CPU |
| SYCL | Intel and Nvidia GPU |
| CUDA | Nvidia GPU |
| HIP | AMD GPU |
| Vulkan | Generic GPU |

*And many others including MUSA, ZenDNN, CANN, OpenCL, VirtGPU, etc.*

---

## 📖 Ecosystem

- The `llama.cpp` project is the main playground for developing new features for the `ggml` library.
- Multimodal support arrived in `llama-server`.
- VS Code & Neovim plugins for FIM completions available.
- Supported by Hugging Face Inference Endpoints.
