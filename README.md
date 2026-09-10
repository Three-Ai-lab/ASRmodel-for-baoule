# omniASR-CTC-300M-baoulé

Fine-tuning de [`facebook/omniASR-CTC-300M`](https://huggingface.co/facebook/omniASR-CTC-300M) (Meta, architecture Wav2Vec2 + tête CTC, ~300M paramètres) pour la reconnaissance vocale automatique (ASR) en **baoulé** (`bci`, Côte d'Ivoire), avec la recette CTC officielle Meta / [fairseq2](https://github.com/facebookresearch/fairseq2), puis conversion vers le format natif [🤗 transformers](https://github.com/huggingface/transformers) pour un usage simplifié.

## Sommaire

- [Aperçu](#aperçu)
- [Données](#données)
- [Entraînement](#entraînement)
- [Résultats](#résultats)
- [Modèles publiés](#modèles-publiés)
- [Utilisation](#utilisation)
  - [Avec 🤗 transformers (recommandé)](#avec--transformers-recommandé)
  - [Avec fairseq2 / omnilingual-asr (format d'origine)](#avec-fairseq2--omnilingual-asr-format-dorigine)
- [Reproduire l'entraînement](#reproduire-lentraînement)
- [Limites connues](#limites-connues)
- [Remerciements](#remerciements)
- [Licence](#licence)

## Aperçu

- **Modèle de base** : `omniASR_CTC_300M` (Meta Omnilingual ASR), encodeur Wav2Vec2 + projection CTC linéaire, 1600+ langues au départ.
- **Langue cible** : baoulé (`bci_Latn`), langue peu dotée parlée en Côte d'Ivoire.
- **Tokenizer** : `omniASR_tokenizer_v1` (SentencePiece), inchangé — pas de ré-entraînement du vocabulaire.
- **Tâche** : ASR CTC (décodage parallèle, sans autorégression).
- **Environnement d'entraînement** : Kaggle (GPU T4, fairseq2 + `omnilingual-asr`).

## Données

- **Source** : [`google/WaxalNLP`](https://huggingface.co/datasets/google/WaxalNLP), configuration `bau_tts`.
- **Filtrage audio** : seuls les enregistrements de **2 à 60 secondes** sont conservés (les plus longs nécessiteraient un découpage aligné sur des timestamps, non disponibles).
- **Prétraitement texte** : suppression des annotations entre crochets non prononcées, normalisation Unicode (NFC), minuscules, conservation des lettres/accents/apostrophes du baoulé, suppression des mots purement numériques.
- **Format** : audio converti en FLAC mono 16 kHz, stocké au format officiel Meta `MixtureParquet`.

## Entraînement

Deux expériences successives, avec la recette CTC officielle (`workflows.recipes.wav2vec2.asr`) :

| Paramètre | Valeur |
|---|---|
| Learning rate | 1e-5 |
| Accumulation de gradient | 4 batches |
| Précision mixte | float16 |
| Activation checkpointing | layerwise |
| Gradient clipping | max_grad_norm = 1.0 |
| Seed | 42 |
| Fine-tuning | modèle complet (tous les paramètres, encodeur non gelé) |

- **Expérience 1** : 500 pas — sert de référence.
- **Expérience 2** : 1000 pas — repart du modèle Meta original (le checkpoint à 500 pas ne contient que les poids, pas l'état de l'optimiseur, donc pas de reprise exacte possible).
- Évaluation sur le split `dev` toutes les 100 pas ; meilleur checkpoint retenu selon le WER.
- Split `test` réservé à l'évaluation finale, jamais vu pendant l'entraînement.

## Résultats

| Version | Pas | WER (dev) | UER/CER (dev) |
|---|---|---|---|
| Baseline (zéro-shot, avant fine-tuning) | 0 | — | — |
| omniASR-CTC-300M-baoulé | 500 | 36.76 % | 11.89 % |
| **omniASR-CTC-300M-baoulé-1000steps** | **1000** | **35.15 %** | **10.95 %** |

La version **1000 pas** est le meilleur checkpoint obtenu et le modèle recommandé pour l'usage en production.

## Modèles publiés

| Version | Dépôt Hugging Face | Formats disponibles |
|---|---|---|
| 500 pas | [`Tree-AI-lab/omniASR-CTC-300M-baoule`](https://huggingface.co/Tree-AI-lab/omniASR-CTC-300M-baoule) | fairseq2 (`model.pt`) |
| **1000 pas** (meilleur) | [`Tree-AI-lab/omniASR-CTC-300M-baoule-1000steps`](https://huggingface.co/Tree-AI-lab/omniASR-CTC-300M-baoule-1000steps) | fairseq2 (`model.pt`) **+** 🤗 transformers natif (`config.json`, `model.safetensors`, tokenizer) |

Les dépôts sont privés par défaut (`PRIVATE_REPO = True`).

## Utilisation

### Avec 🤗 transformers (recommandé)

Le checkpoint 1000 pas a été converti en `Wav2Vec2ForCTC` natif — aucune dépendance à fairseq2 n'est nécessaire.

```bash
pip install transformers torch torchaudio soundfile
```

**Le plus simple — `pipeline()` :**

```python
import soundfile as sf
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="Tree-AI-lab/omniASR-CTC-300M-baoule-1000steps",
    device=0,  # GPU si disponible, sinon -1 (CPU)
)

# Charger l'audio manuellement (évite les soucis de décodage type torchcodec/FFmpeg)
waveform, sr = sf.read("audio.wav", dtype="float32", always_2d=True)
waveform = waveform.mean(axis=1)  # stéréo -> mono

result = pipe({"array": waveform, "sampling_rate": sr})
print(result["text"])
```

**Contrôle manuel :**

```python
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL_ID = "Tree-AI-lab/omniASR-CTC-300M-baoule-1000steps"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).eval()

waveform, sr = torchaudio.load("audio.wav")
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
if sr != 16_000:
    waveform = torchaudio.functional.resample(waveform, sr, 16_000)

inputs = processor(waveform.squeeze().numpy(), sampling_rate=16_000, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

pred_ids = torch.argmax(logits, dim=-1)
print(processor.batch_decode(pred_ids)[0])
```

**Audios longs (> 40 s)** — découpage automatique avec chevauchement :

```python
pipe = pipeline(
    "automatic-speech-recognition",
    model="Tree-AI-lab/omniASR-CTC-300M-baoule-1000steps",
    chunk_length_s=35,
    stride_length_s=2,
)
```

> La sortie est en minuscules, sans ponctuation — comportement normal du tokenizer `omniASR_tokenizer_v1` (modélise uniquement les caractères parlés et les tons, pas la casse ni la ponctuation).

### Avec fairseq2 / omnilingual-asr (format d'origine)

```bash
git clone --depth 1 https://github.com/facebookresearch/omnilingual-asr.git
pip install -e "omnilingual-asr[data]"
```

```python
import torch
from fairseq2.data.tokenizers.hub import load_tokenizer
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

pipeline = ASRInferencePipeline(model_card="omniASR_CTC_300M", device="cuda", dtype=torch.float16)

state_dict = torch.load("model.pt", map_location="cpu", weights_only=True)  # téléchargé depuis le dépôt HF
pipeline.model.load_state_dict(state_dict, strict=True)
pipeline.model.eval()

transcripts = pipeline.transcribe(["audio.wav"], batch_size=1)
print(transcripts[0])
```

## Reproduire l'entraînement

Le notebook complet (`finetuning-asr-baoulé-300M-officiel.ipynb`) couvre, dans l'ordre :

1. Installation de `omnilingual-asr` + fairseq2 (Kaggle, GPU + Internet requis).
2. Préparation du dataset WaxalNLP au format `MixtureParquet`.
3. Vérification du dataloader.
4. Mesure de la baseline zéro-shot sur `dev`.
5. Entraînement CTC — 500 pas, puis 1000 pas.
6. Évaluation sur 10 audios du split `test` (WER/CER) + test sur audio personnel.
7. Publication des checkpoints fairseq2 sur le Hub.
8. **Conversion vers 🤗 transformers** (`Wav2Vec2ForCTC`) avec vérification de parité fairseq2 ↔ HF, puis publication.

La logique de conversion fairseq2 → transformers est adaptée de [ahmedadelattia/omnilingual_to_hf](https://github.com/ahmedadelattia/omnilingual_to_hf).

## Limites connues

- Entraîné sur un sous-ensemble relativement restreint (`bau_tts`, audios de 2 à 60 s uniquement) : le WER reflète ce domaine spécifique et peut varier sur d'autres types d'audio (bruit de fond, accents régionaux, enregistrements spontanés).
- Sortie sans ponctuation ni casse, héritée du tokenizer de base.
- Le pipeline `transformers` (option `chunk_length_s`) peut nécessiter FFmpeg/torchcodec selon l'environnement ; en cas d'erreur de décodage, charger l'audio manuellement avec `soundfile` en entrée (voir exemple ci-dessus).
- Checkpoints sauvegardés en `save_model_only`, sans état d'optimiseur — pas de reprise d'entraînement à l'identique depuis un checkpoint publié.

## Remerciements

- [Meta AI — Omnilingual ASR Team](https://github.com/facebookresearch/omnilingual-asr) pour le modèle de base et la recette d'entraînement CTC.
- [Google — WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP) pour les données `bau_tts`.
- [ahmedadelattia/omnilingual_to_hf](https://github.com/ahmedadelattia/omnilingual_to_hf) pour la logique de conversion fairseq2 → transformers.

## Licence

Le modèle de base `omniASR_CTC_300M` est distribué sous licence Apache-2.0. Se référer aux conditions d'utilisation du dataset `google/WaxalNLP` pour les données d'entraînement.
