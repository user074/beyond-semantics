
# Beyond Semantics

This repository hosts our experiments around restoring **spatial awareness** in VLMs, built on top of **LLaVA**. It includes training add-ons (flags) and small inference hooks (e.g., permutation probes).

## TL;DR

* Set up the conda environment.
* Let's roll!

Hugging Face model zoo: 

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/collections/user074/beyond-semantics)

2DS Dataset: 

[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/user074/2DSyntheticDataset)

Quick start for the diagnostic tools: 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/user074/beyond-semantics/blob/main/diagnostic_tools/Diagnostic_Tools_Walkthrough.ipynb)

Just open the notebook and run the cells if you are just interested in the diagnosis.


---

## 1) Installation

### Create the environment

```bash
# from repo root
conda env create -f environment.yaml
conda activate beyond-semantics
```
Also install the VLM-Visualizer repo:

```bash
git clone https://github.com/zjysteven/VLM-Visualizer.git
```

Then install the LLaVA repo:
```bash
cd LLaVA
pip install -e .
cd ..
```


---

## 2) Checkpoints you’ll need

We place model and projector weights under local folders for reproducibility (you can also rely on Hugging Face cache if you prefer).

**Models**

* `openai/clip-vit-large-patch14-336`
* `liuhaotian/llava-v1.5-7b`
Also corresponding models we trained for diagnostic probes are available in the model zoo.

**Projector**

* `liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5`

A typical layout:

```
beyond-semantics
├── LLaVA
├── model
│   ├── lmsys/vicuna-7b-v1.5
│   ├── openai/clip-vit-large-patch14-336
│   ├── liuhaotian/llava-v1.5-7b
│   └── projector
        └── liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5
```


---

## 3) Training (optional)

Training follows the **original LLaVA** recipes. Use the scripts and flags described in `LLaVA/` (see here [LLaVA/README.md](LLaVA/README.md) for our add-ons like projector normalization, multi-layer features, compression, etc.).


---

## 4) Diagnoistic probes

Detailed instructions are available in the [diagnostic_tools/README.md](diagnostic_tools/README.md).

### Permutation probe at inference
---


We expose a simple permutation hook that shuffles **vision token order** inside `model.generate()` to test permutation sensitivity.

### Code snippet (where the generate call happens)

```python
output_ids = model.generate(
    input_ids=input_ids,
    images=image_tensor,
    image_sizes=[image_size],
    do_sample=(args.temperature > 0),
    temperature=args.temperature,
    max_new_tokens=args.max_new_tokens,
    streamer=streamer,
    use_cache=False,              # keep False if your permutation logic assumes fresh KV each step
    permutation=args.permutation  # <-- toggle with a CLI flag
)
```

We provide an example in `LLaVA/llava/serve/cli.py` that accepts a `--permutation` flag so you can try the probe from the command line.

### Example command

```bash
python -m llava.serve.cli \
  --model-path liuhaotian/llava-v1.5-7b \
  --image-file "https://llava-vl.github.io/static/images/view.jpg" \
  --load-4bit \
  --permutation
```
---
### Rest of the diagnostic tools

The rest of the diagnostic tools are located in the [diagnostic_tools/](diagnostic_tools/) directory.
Including:
- Vision vs Text Token norms analysis
- RoPE sensitivity probe
- System prompt attention shares
- CMB analysis
- Hidden norms analysis
- Attention heatmaps


---

## 5) Evaluation

Evaluation details are available in the [evaluation/eval_instruction.md](evaluation/eval_instruction.md).


## 6) Citation
If you find this work useful, please cite:

```bibtex
@misc{qi2025semantics,
    title={Beyond Semantics: Rediscovering Spatial Awareness in Vision-Language Models},
    author={Jianing Qi and Jiawei Liu and Hao Tang and Zhigang Zhu},
    year={2025},
    eprint={2503.17349},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```