# MoA-MultiAgent-Phi2-LLM-Medical-AI

## Abstract
Accurate and interpretable diagnostic support continues to pose significant challenges for clinical artificial intelligence. This paper **proposes MoA-Med**, a computationally efficient **Mixture-of-Agents architecture**, wherein multiple 4-bit quantized, **LoRA-finetuned Phi-2 language models** collaboratively assign ICD-10 and SNOMED codes from synthetic electronic health records (EHRs) generated using Synthea. Compared to a single finetuned Phi-2 agent, MoA-Med achieves a relative macro-F1 improvement of +14 pp, with optimal performance obtained using four domain-specialist agents. Additionally, MoA-Med maintains inference latency below 0.7 seconds and operates within an 8 GB GPU memory constraint. Experimental results demonstrate that parameter-efficient domain specialization, combined with a lightweight rule-based routing and consensus mechanism, **yields consistent but moderate improvements in diagnostic accuracy and efficiency**, making it suitable for deployment in computationally constrained environments. 

---

## Project Objectives
1. **Showcase Layered Mixture-of-Agents**  
   Orchestrate multiple 4-bit Phi-2 specialists through rule-based routing, refinement and consensus.
2. **Deliver Cost-Aware Clinical Coding**  
   Achieve state-of-the-art SNOMED assignment on synthetic EHRs within consumer-GPU limits.
3. **Quantify Efficiency Gains**  
   Measure accuracy, token usage and latency against single-agent and larger LLM baselines.
4. **Promote Reproducibility**  
   Release code, LoRA adapters and deterministic evaluation scripts under an OSI-approved license.

---

## System Requirements & Dataset

### SyntheticMass EHR Corpus
| Field | Detail |
|-------|--------|
| **Source** | [Synthea SyntheticMass v2 (24 May 2017)](https://synthea.mitre.org/downloads) |
| **Volume** | 21 GB (1.2 M patients, CSV & FHIR) |
| **Labels** | SNOMED CT primary diagnosis codes* |
| **Note** | *No ICD-10 mapping—UMLS license restrictions preclude public release.*

### Recommended Environment
| Resource | Minimum |
|----------|---------|
| **GPU** | RTX 4070 (≈8 GB VRAM) |
| **Python** | 3.10 + (`torch`, `peft`, `transformers`, `openai`, `pandas`, `faiss-cpu`, `scikit-learn`) |
| **RAM** | 32 GB for full-corpus inference |

*See `environment.yml` for specific package versions used in this project.*

---

## Architecture & Workflow

<div align="center">

![MoA Architecture](assets/Model%20Architecture.jpg)

</div>

*Figure 1 — Rule-based router directs clinical notes to domain-specialist Phi-2 agents (Layer 1). Outputs pass to a GPT-3.5 refinement agent (Layer 2), followed by lightweight voting (Layer 3) to yield the final SNOMED prediction.*

1. **Layer 0 – Prompt Router**  
   Routes incoming clinical notes into specialized queues: *cardiology*, *metabolic*, *respiratory_ent*, or *generalist*.
2. **Layer 1 – Domain Specialists**  
   Four distinct 4-bit LoRA-finetuned Phi-2 models independently generate SNOMED-based diagnostic hypotheses.
3. **Layer 2 – GPT-3.5 Optimizer**  
   Consolidates outputs from domain specialists, refines clinical reasoning, and produces a ranked list of ICD-10 codes.
4. **Layer 3 – Consensus**  
   Applies majority voting, with ties resolved using token-level probability scores for confidence assessment.

---

## Methodological Framework

1. **Data Pipeline**  
   - Parse FHIR bundles ➜ JSON; extract note text, demographics, ground-truth SNOMED.  
   - Split into train/validation/test (80/10/10).
2. **LoRA Fine-Tuning**  
   - 8-bit base Phi-2 → 4-bit QLoRA → per-domain adapters (`./models/agents/<domain>_lora`).  
   - Training: 3 epochs, batch 8, LR 2e-4, rank 8 adapters.
3. **Inference Loop** (`src/evaluate.py`)  
   - `--agent cardio` loads correct adapter via `peft.PeftModel.from_pretrained`.  
   - Formats patient context ➜ `model.generate(...)`; parses JSON-like SNOMED output.  
4. **Consensus Logic** (`src/consensus.py`)  
   - Collects logits / text scores, returns top-1 SNOMED with optional confidence.

---

## Evaluation & Results

| Model Variant | Macro-F1 | Accuracy | Tokens / case | Latency (ms) |
|---------------|----------|----------|---------------|--------------|
| **Single Phi-2** | 0.42 | 0.55 | 570 | 430 |
| **MoA-Med (ours)** | **0.64** | **0.79** | 780 | 890 |
| **GPT-4 (33 B)** | 0.67 | 0.81 | 2 350 | 3 900 |

---

## Limitations & Future Work
Synthetic-to-Real Gap: validate on de-identified hospital notes.

Class Imbalance: explore focal-loss fine-tuning for rare SNOMED codes.

Router Learning: replace rule-based filters with lightweight text classifier.

Confidence Calibration: add temperature scaling to consensus logits.

---

## References

1. Wang, Junlin, Jue Wang, Ben Athiwaratkun, Ce Zhang, and James Zou. ["Mixture-of-Agents Enhances Large Language Model Capabilities"](https://arxiv.org/abs/2406.04692). *arXiv preprint arXiv:2406.04692*, 2024.

2. Wu, Yixuan, Yutong He, Yuxiao Jiang, Yuancheng Liu, and Yang Yang. ["PMC-LLaMA: Further Finetuning LLaMA on Medical Papers"](https://arxiv.org/abs/2304.14454). *arXiv preprint arXiv:2304.14454*, 2023.

3. Ji, Ke, Jiahao Xu, Tian Liang, Qiuzhi Liu, Zhiwei He, Xingyu Chen, Xiaoyuan Liu, Zhijie Wang, Junying Chen, Benyou Wang, Zhaopeng Tu, and Haitao Mi. ["The First Few Tokens Are All You Need: An Efficient and Effective Unsupervised Prefix Fine-Tuning Method for Reasoning Models"](https://arxiv.org/abs/2503.02875). *arXiv preprint arXiv:2503.02875*, 2025.

4. Yang, Xintian, Tongxin Li, Han Wang, et al. ["Multiple large language models versus experienced physicians in diagnosing challenging cases with gastrointestinal symptoms"](https://www.nature.com/articles/s41746-025-01486-5). *npj Digital Medicine* **8**, 85 (2025).

5. Jiang, Songtao, Tuo Zheng, Yan Zhang, Yeying Jin, Li Yuan, and Zuozhu Liu. ["Med-MoE: Mixture of Domain-Specific Experts for Lightweight Medical Vision-Language Models"](https://arxiv.org/abs/2404.10237). *arXiv preprint arXiv:2404.10237*, 2024.

6. Yu, Hongzhou, Tianhao Cheng, Ying Cheng, and Rui Feng. ["FineMedLM-o1: Enhancing the Medical Reasoning Ability of LLM from Supervised Fine-Tuning to Test-Time Training"](https://arxiv.org/abs/2501.09213). *arXiv preprint arXiv:2501.09213*, 2025.

7. Wang, Yizhong, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. ["Self-Instruct: Aligning Language Models with Self-Generated Instructions"](https://arxiv.org/pdf/2212.10560). *arXiv preprint arXiv:2212.10560*, 2023.

---

## License
**Apache License 2.0** or **MIT License** recommended for broad academic and industry adoption.
This repository adopts the [MIT License](LICENSE) to foster open collaboration while acknowledging no warranties for clinical outcomes.
