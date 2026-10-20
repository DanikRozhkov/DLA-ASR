"""
Trains a SentencePiece tokenizer on the transcripts of the training set
so that model can predict subwords instead of single characters
"""

import json
from pathlib import Path

import hydra


def _write_corpus(config, corpus_path):
    """
    Pulls every transcript out of the already downloaded dataset index
    and writes them to a single text file for the tokenizer to train on

    Args:
        config (DictConfig): Hydra config with the train_bpe.data_dir
            and train_bpe.parts settings
        corpus_path (Path): file to write the normalized transcripts to
    """
    from src.text_encoder.ctc_text_encoder import CTCTextEncoder

    data_dir = Path(config.train_bpe.data_dir)
    lines = []
    for part in config.train_bpe.parts:
        index_path = data_dir / f"{part}_index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"{index_path} not found - run the normal training/data "
                f"pipeline first so this part is downloaded and indexed"
            )
        with index_path.open() as f:
            index = json.load(f)
        for entry in index:
            normalized = CTCTextEncoder.normalize_text(entry["text"])
            if normalized.strip():
                lines.append(normalized)

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text("\n".join(lines))
    print(f"Wrote {len(lines)} normalized transcript lines -> '{corpus_path}'")


@hydra.main(version_base=None, config_path="../configs", config_name="train_bpe")
def main(config):
    """
    Builds the training corpus if needed and trains a new SentencePiece
    tokenizer from it

    Args:
        config (DictConfig): Hydra config with the train_bpe settings,
            such as corpus_path, model_prefix, and vocab_size
    """
    import sentencepiece as spm

    corpus_path = Path(config.train_bpe.corpus_path)
    if config.train_bpe.rebuild_corpus or not corpus_path.exists():
        _write_corpus(config, corpus_path)
    else:
        print(f"Reusing existing corpus at '{corpus_path}'")

    model_prefix = config.train_bpe.model_prefix
    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=config.train_bpe.vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
    )
    print(f"\nTrained BPE tokenizer -> '{model_prefix}.model' / '.vocab'")
    print(
        f'Set text_encoder.sp_model_path: "{model_prefix}.model" in your training config'
    )


if __name__ == "__main__":
    main()
