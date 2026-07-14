# Evaluation of Energy Consumption and Performance in Large Language Models (LLMs)

Benchmarking system and experimental data for my MSc dissertation at the University of Coimbra (Informatics Engineering, Artificial Intelligence). The goal is to measure how much energy LLMs use during inference, and to see how prompt language, prompt difficulty, model architecture, and the hardware itself affect that energy, on GPUs outside the data-center scale.

## Repository layout

```
benchmark/     model installer and benchmark runner
analysis/      per-run and cross-run analysis scripts
figures/       result figures for the two GPUs used
requirements.txt
```

## Background

For a deployed model, inference runs constantly while training happens once, so over time inference ends up dominating the energy cost. The problem is that existing studies are hard to compare, because they use different hardware, different workloads, and measure different things. This project is an attempt at a setup that is controlled enough to compare results across models and languages.

What it does:

- measures net CPU and GPU energy per prompt, subtracting an idle baseline
- runs the same prompts translated into 8 languages, split into easy/medium/hard tiers
- covers 8 models across 4 families and 3 size ranges, including both dense and Mixture-of-Experts models
- reports energy per token as the main number, alongside latency, throughput and timeout rate

## Hardware used

| GPU | VRAM | Architecture | Environment | What was measured |
|---|---|---|---|---|
| RTX A4000 | 16 GB | Ampere | bare-metal | CPU + GPU |
| RTX PRO 6000 Blackwell Max-Q | 96 GB | Blackwell | Xen VM, GPU passthrough | GPU only |

The A4000 runs on bare metal so it can report CPU power. The Blackwell has enough memory to fit every model, including the 32B and 120B ones, but it runs inside a VM that does not expose CPU power sensors.

## Running it

Set up the environment and build llama.cpp:

```bash
cd benchmark
bash setup.sh
```

Install the models (downloads, converts to GGUF, quantises, checks VRAM):

```bash
bash install_models.sh
```

Run the benchmark:

```bash
bash benchmark.sh
# or use the GUI:
python3 benchmark_gui.py
```

Settings like model paths, timeouts, run mode and repetition count are in `benchmark/benchmark_config.sh`.

Analyse a results folder:

```bash
cd analysis
bash rerun_analysis.sh /path/to/results_session
```

One thing to watch: keep spaces and parentheses out of the results folder name. A folder like `2026-06-24_21-07-41 (2)` breaks the analysis because the path gets split on the space. Rename it to something like `2026-06-24_21-07-41_v2` first.

## Metrics

- energy per token (mWh/token) - the main efficiency number, net GPU energy divided by generated tokens
- energy per prompt (mWh)
- throughput (tokens/s) and latency (s)
- timeout rate - share of prompts that ran past the per-prompt time limit
- effective energy per attempted prompt - total energy over every prompt attempted, not just the ones that finished. This matters because a model that times out on hard prompts otherwise looks artificially cheap, since it is only scored on the prompts it completed.

## Notes on reproducibility

Every run subtracts a measured idle baseline and starts with a warm-up prompt, and the whole thing is repeated over several runs. Consistency across runs is checked with the coefficient of variation, and outliers within each model/language/difficulty group are flagged with the IQR rule.

## Models

Gemma 3 4B, Qwen 2.5 7B, Qwen 2.5 Coder 7B, Mistral 7B, Qwen 3 32B, Qwen 2.5 Coder 32B, GPT-OSS 20B (MoE), GPT-OSS 120B (MoE).

