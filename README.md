# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains a solution for Automatic Speech Recognition task based on DeepSpeech2 architecture, trained on LibriSpeech dataset. Full report can be seen [here](https://www.comet.com/daniel-rozhkov/pytorch-template-asr-example/reports/x7pJ5mPVAHbZtoA1asEpRCEPj)

## Results

During all runs augmentation was used: frequency masking, time masking, gain, noise injection, pitch shift

### [Char-level baseline (100 epochs, train-clean-100+360)](https://www.comet.com/daniel-rozhkov/pytorch-template-asr-example/ee8eebbba24a49dbb858c997f7fb8f2c)
Used 4-gram KenLM with parameters:  beam width 10, alpha=0.6, beta=2.0

| Partition | Variant | CER | WER |
|---|---|---|---|
| val | Argmax | 0.1641 | 0.4311 |
| val | BeamSearch | 0.1608 | 0.4252 |
| test | Argmax | 0.1514 | 0.4096 |
| test | BeamSearch | 0.1478 | 0.4028 |
| test | BeamSearchLM | 0.1368 | 0.3235 |

### [Final model: BPE + warm start (70 epochs, encoder warm-started from the char baseline)](https://www.comet.com/daniel-rozhkov/pytorch-template-asr-example/44pylo85kc4m31b27kxina9967zdogb3)
Used 4-gram KenLM with parameters:  beam width 30, alpha=0.6, beta=2.0

| Partition | Variant | CER | WER |
|---|---|---|---|
| val | Argmax | 0.1437 | 0.3225 |
| val | BeamSearch | 0.1422 | 0.3191 |
| test | Argmax | 0.1318 | 0.3033 |
| test | BeamSearch | 0.1303 | 0.2998 |
| test | BeamSearchLM | 0.1071 | 0.2303 |

Pushing beam width to 50 and switching to a 5-gram KenLM model further improved the final result:
| Beam width | LM | test CER | test WER |
|---|---|---|---|
| 30 | 4-gram | 0.1071 | 0.2303 |
| 50 | 4-gram | 0.1053 | 0.2265 |
| 50 | 5-gram | **0.1044** | **0.2237** |

Implemented bonuses: LM, BPE

## Installation

Follow these steps to install the project:

0. Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)

   a. `conda` version:

   ```bash
   conda create -n project_env python=3.10

   # activate env
   conda activate project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

3. To download model checkpoints from Hugging Face use the script `download_resources.sh`

4. Install system dependencies required to build kenlm
   ```bash
   # Ubuntu
   sudo apt-get install -y cmake build-essential libboost-all-dev libeigen3-dev
   ```

## How To Use

### Training
To train a model, run the following command:

```bash
python train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```
Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

### Inference
To run inference (evaluate the model and save predictions):

```bash
python inference.py \
  +datasets.custom.audio_dir=/path/to/audio \
  +datasets.custom.transcription_dir=/path/to/transcriptions
```
Predictions are decoded with beam search and language model fusion and saved to data/saved

### Evaluation (computing WER/CER)
To score predictions against ground truth separately(!) from inference.py:
```bash
python calculate_wer_cer.py \
  --predictions_dir data/saved/custom_dir_predictions/custom \
  --ground_truth_dir /path/to/transcriptions
```

### Language model tuning
To re-tune the shallow fusion weights alpha/beta for a given checkpoint, or to re-run the full Argmax/BeamSearch/BeamSearchLM evaluation:
```bash
python -m src.text_encoder.lm.lm_tuning checkpoint_path=saved/.../model_best.pth
python -m src.text_encoder.lm.lm_eval checkpoint_path=saved/.../model_best.pth
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
