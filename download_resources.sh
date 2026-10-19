#!/usr/bin/env bash
# Downloads the final model checkpoints, BPE tokenizer, and the KenLM
# language model used for shallow fusion. Run from the repository root:
# bash download_resources.sh
set -euo pipefail

BASE_URL="https://huggingface.co/DanikRozhkov/dla-asr/resolve/main"

mkdir -p saved/deepspeech2_bpe500_clean460_70ep
mkdir -p saved/deepspeech2_librispeech_clean460_100ep
mkdir -p bpe
mkdir -p lm

echo "Downloading final BPE model checkpoint..."
curl -L -o saved/deepspeech2_bpe500_clean460_70ep/model_best.pth \
  "$BASE_URL/bpe_model.pth"

echo "Downloading char-level baseline checkpoint (used as warm-start source)..."
curl -L -o saved/deepspeech2_librispeech_clean460_100ep/model_best.pth \
  "$BASE_URL/char_model.pth"

echo "Downloading BPE tokenizer..."
curl -L -o bpe/librispeech_bpe500.model \
  "$BASE_URL/librispeech_bpe500.model"

echo "Downloading 4-gram KenLM binary..."
curl -L -o lm/4-gram.binary \
  "$BASE_URL/4-gram.binary"

echo "Done"
