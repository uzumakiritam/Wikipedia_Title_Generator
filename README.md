# Wikipedia Title Generation Using Sequence-to-Sequence Models

## Overview
This project explores abstractive text summarization for generating concise Wikipedia-style titles using **sequence-to-sequence (seq2seq)** models.  
It compares traditional **RNN-based encoder-decoder architectures** with modern **Transformer-based T5 models**, analyzing trade-offs in performance, efficiency, and decoding strategies.

---

## Objectives
- Build a preprocessing pipeline for noise-free, semantically consistent text.  
- Implement and compare RNN-based and Transformer-based title generation models.  
- Optimize training using mixed precision, gradient clipping, and learning rate scheduling.  
- Evaluate results using ROUGE metrics for content overlap and precision.

---

## Preprocessing Pipeline
Text cleaning and normalization steps:
- Lowercasing and removal of unwanted symbols using regex.  
- Tokenization with NLTK.  
- Minimal stopword removal for balance between brevity and meaning.  
- Lemmatization using WordNetLemmatizer.  
- Vocabulary built with threshold frequency (min count 19), resulting in ~46,000 tokens.

---

## Model Architectures

### RNN-Based Seq2Seq
Implemented four configurations:
1. **Basic RNN + Greedy Search**  
2. **Basic RNN + Beam Search**  
3. **Hierarchical Encoder + Dual Decoder + Greedy Search**  
4. **Hierarchical Encoder + Dual Decoder + Beam Search**

All models used GRU-based encoders/decoders with dropout for regularization.

### Transformer-Based T5
1. **Fine-Tuned T5-small** — trained on Wikipedia-like text.  
2. **Prompt-Based Flan-T5 (Base & Large)** — tested in zero-shot mode using handcrafted prompts.

---

## Training Optimizations
- **Mixed Precision** for faster training and reduced GPU memory usage.  
- **Gradient Clipping** to stabilize large gradients.  
- **Early Stopping** on validation loss plateau.  
- **Dynamic Learning Rate Scheduling** for convergence control.  
- CUDA memory optimization using `torch.cuda.empty_cache()` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

---

## Evaluation

### Metrics
ROUGE-1, ROUGE-2, and ROUGE-L were used to measure n-gram overlap between generated and reference titles.

### Results Summary

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|--------|----------|----------|----------|
| Basic RNN + Greedy | 0.2393 | 0.0535 | 0.2393 |
| Basic RNN + Beam | 0.2395 | 0.0477 | 0.2395 |
| Hierarchical RNN + Greedy | 0.1330 | 0.0238 | 0.1330 |
| Hierarchical RNN + Beam | 0.1291 | 0.0202 | 0.1291 |
| Fine-Tuned T5 (Greedy) | **0.8780** | **0.6778** | **0.8780** |
| Fine-Tuned T5 (Beam) | 0.8719 | 0.6678 | 0.8719 |
| Flan-T5-Base (“Generate a title...”) | 0.7547 | 0.5321 | 0.7547 |
| Flan-T5-Large (“Appropriate title...”) | 0.7649 | 0.5603 | 0.7649 |

---

## Observations
- **T5-small (fine-tuned)** achieved the highest overall ROUGE scores.  
- **Greedy decoding** often outperformed beam search for short title tasks.  
- **Flan-T5 models** performed well in zero-shot settings with strong prompt engineering.  
- Beam search tended to favor fluency but slightly reduced bigram overlap (ROUGE-2).

---

## Execution Notes
- RNN models trained for ~3000–3500 seconds depending on configuration.  
- T5-small training completed in ~1700 seconds.  
- Flan-T5 inference ranged from 90–200 seconds depending on model size.

---

## Conclusion
Seq2Seq-based title generation demonstrates that:
- RNNs with hierarchical encoders can handle longer sequences efficiently.  
- Fine-tuned T5 models provide state-of-the-art performance with minimal decoding complexity.  
- Prompt-based Flan-T5 offers an effective zero-shot alternative for lightweight deployment.  

---

