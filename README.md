# Fugue State – Soundscape of Forgetting with Neural Ablation

> A creative soundscape built from a custom-trained DCGAN, progressively “forgetting” its learned representations of guitar chords through neural ablation, and reconstructing the resulting mel-spectrograms into audio.

**Full project write-up:** [Read on my portfolio](https://samirsfolder.com)  
**Demo video:** [Watch on YouTube](https://www.youtube.com/watch?v=pBVwEdPIxIE)

---

## Overview
*Fugue State* explores **network bending in GAN generators via neural ablation** as a creative framework for sonic memory loss.

I developed and trained a DCGAN from scratch to map 100-dimensional latent vectors to **128×128 mel-spectrograms** of guitar chords.  
An automated pipeline progressively zeros out (ablates) channels in the generator’s deconvolution layers.  
Each ablated spectrogram is inverted back to audio via a tuned **Griffin–Lim** algorithm, then stitched into a continuous soundscape with overlap and reverb.

The result is an evolving audio piece where harmonic clarity fades, timbres distort, and noise emerges — a sonic metaphor for gradual forgetting.

---

## Concept & Inspiration
The project draws inspiration from *What I Saw Before the Darkness* (Cole, 2019), in which a portrait GAN’s neurons are switched off one by one until faces dissolve into abstraction.  
I wanted to translate that visual metaphor into sound, exploring:

- **Anthropomorphic machine forgetting** — while the network has no true memory, its sonic degradation can evoke emotional responses like nostalgia or unease.
- **Material fragility of AI** — revealing how generative systems can fracture under controlled interventions.
- **Dialogues with memory/decay works** — such as Leyland Kirby’s *Everywhere at the End of Time* and William Basinski’s *The Disintegration Loops*.

By foregrounding ablation as an *artistic process*, *Fugue State* treats collapse not as failure, but as compositional material.

---

## Features
- **Custom DCGAN** trained on guitar chords dataset (no transfer learning).
- **Progressive channel ablation** in generator layers.
- **Automated spectrogram inversion** with tuned Griffin–Lim parameters.
- **End-to-end pipeline** from dataset to soundscape.
- **Creative sound design** via decay, overlap, and reverb processing.
- **Visualization of neuron death** alongside audio playback.

---

## Tech Stack
- **Language & Frameworks:** Python, PyTorch, Jupyter Notebooks
- **ML:** DCGAN architecture, spectral normalization, hinge loss, mel-spectrogram loss
- **DSP:** Griffin–Lim inversion, reverb, segment overlap
- **Dataset:** [Guitar Chords V2](https://www.kaggle.com/datasets/fabianavinci/guitar-chords-v2)
- **Environment:** Conda + pip requirements

---

## How to Run

1. **Download**
   - Download the project and the data from the dataset link above.
   - Place the `data` folder at the root of the project.

2. **Setup Environment**
   ```bash
   conda create -n fugue python=3.10
   conda activate fugue
   pip install -r requirements.txt
   ```

3. **Run Pipeline**
   - Open `notebooks/demo.ipynb`.
   - Step through:
     1. Data preview  
     2. Model training (optional, pretrained checkpoint provided)  
     3. Sample generation  
     4. Ablation & soundscape synthesis  
     5. Neuron-death visualization  

---

## File Structure
```
/src
  data/dataset.py       # PyTorch dataset for chord spectrograms
  models.py             # DCGAN generator & discriminator
  utils.py              # Loss functions & audio inversion
  train.py              # Training loop

/notebooks
  demo.ipynb            # Full pipeline walkthrough

```
---

## Results

**Soundscape (stitched & ablated):** [Video Demo](https://www.youtube.com/watch?v=pBVwEdPIxIE)

---

## Limitations & Future Work
- **Spectrogram inversion** remains the fidelity bottleneck — consider neural vocoder replacement.
- **Segment length** (~1.5s) constrains expressive decay — extend receptive field.
- **Ablation targeting** could evolve into adaptive or reversible scheduling.
- Potential for **interactive control**, letting a performer “play” the model’s memory.

---

## References
- Radford, A., Metz, L., & Chintala, S. (2016) *Unsupervised Representation Learning with DCGANs*
- Griffin, D., & Lim, J. (1984) *Signal estimation from modified STFT*
- Cole, S. (2019) *Watching AI Slowly Forget a Human Face*, Vice
- Kirby, L. (2019) *Everywhere at the End of Time*
- Basinski, W. (2002) *The Disintegration Loops*
