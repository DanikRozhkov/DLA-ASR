"""
Fast, repeatable evaluation of a checkpoint with fixed LM-fusion alpha/beta,
reporting Argmax / BeamSearch / BeamSearchLM CER and WER on val and test.

Meant to be rerun cheaply every time the acoustic model improves (longer
training, architecture tweaks, ...) without touching the training pipeline
at all: the acoustic model runs once per partition (pure inference, no
training step involved -- unlike resuming through train.py with a dummy
epoch, which would apply a real gradient update before evaluating), then
every decode variant is evaluated in parallel across CPU cores.
"""

import json
import multiprocessing as mp
import warnings

import hydra
import torch
from tqdm import tqdm

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder.lm.lm_utils import load_kenlm_model
from src.utils.eval_utils import extract_log_probs
from src.utils.init_utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)

_WORKER = {}


def _init_worker(lm_path, alphabet, sp_model_path, topk, beam_width):
    from src.text_encoder.ctc_text_encoder import CTCTextEncoder

    _WORKER["lm"] = load_kenlm_model(lm_path)
    _WORKER["encoder"] = CTCTextEncoder(
        alphabet=alphabet, sp_model_path=sp_model_path, topk=topk
    )
    _WORKER["beam_width"] = beam_width


def _decode_chunk(task):
    combo_id, use_lm, alpha, beta, chunk = task
    encoder = _WORKER["encoder"]
    beam_width = _WORKER["beam_width"]
    lm = _WORKER["lm"] if use_lm else None

    cer_errors, wer_errors = [], []
    for s in chunk:
        target = encoder.normalize_text(s["text"])
        pred = encoder.ctc_beam_search(
            s["log_probs"], beam_width, lm_model=lm, alpha=alpha, beta=beta
        )
        cer_errors.append(calc_cer(target, pred))
        wer_errors.append(calc_wer(target, pred))
    return combo_id, cer_errors, wer_errors


def _argmax_metrics(encoder, samples):
    cer_errors, wer_errors = [], []
    for s in samples:
        target = encoder.normalize_text(s["text"])
        inds = s["log_probs"].argmax(axis=-1)
        pred = encoder.ctc_decode(inds)
        cer_errors.append(calc_cer(target, pred))
        wer_errors.append(calc_wer(target, pred))
    return sum(cer_errors) / len(cer_errors), sum(wer_errors) / len(wer_errors)


def _chunk(samples, n_chunks):
    n_chunks = max(1, min(n_chunks, len(samples)))
    size = (len(samples) + n_chunks - 1) // n_chunks
    return [samples[i : i + size] for i in range(0, len(samples), size)]


@hydra.main(version_base=None, config_path="../../configs", config_name="lm_eval")
def main(config):
    set_random_seed(config.trainer.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.trainer.device != "auto":
        device = config.trainer.device

    alpha = config.lm_eval.alpha
    beta = config.lm_eval.beta
    beam_width = config.lm_eval.beam_width
    lm_path = config.lm_eval.lm_path
    n_workers = config.lm_eval.n_workers or mp.cpu_count()
    partitions = list(config.lm_eval.partitions)

    from src.text_encoder.ctc_text_encoder import CTCTextEncoder

    sp_model_path = config.text_encoder.get("sp_model_path")
    topk = config.text_encoder.get("topk", 45)
    encoder_for_argmax = CTCTextEncoder(
        alphabet=config.text_encoder.get("alphabet"),
        sp_model_path=sp_model_path,
        topk=topk,
    )

    samples_by_partition = {}
    argmax_results = {}
    for part in partitions:
        samples = extract_log_probs(config, part, device)
        samples_by_partition[part] = samples
        cer, wer = _argmax_metrics(encoder_for_argmax, samples)
        argmax_results[part] = {"cer": cer, "wer": wer}

    # Build decode tasks: (partition, variant) x N chunks, flattened across
    # all combos so the pool balances load over every available core
    # instead of assigning at most one combo per worker.
    tasks = []
    combo_names = {}
    for part in partitions:
        samples = samples_by_partition[part]
        chunks = _chunk(samples, n_workers)
        for variant, use_lm, a, b in [
            ("BeamSearch", False, 0.0, 0.0),
            ("BeamSearchLM", True, alpha, beta),
        ]:
            combo_id = f"{part}:{variant}"
            combo_names[combo_id] = (part, variant)
            for chunk in chunks:
                tasks.append((combo_id, use_lm, a, b, chunk))

    print(
        f"Decoding {len(tasks)} chunks across {len(combo_names)} "
        f"(partition, variant) combos using {n_workers} worker processes"
    )

    combo_cer = {k: [] for k in combo_names}
    combo_wer = {k: [] for k in combo_names}
    with mp.Pool(
        n_workers,
        initializer=_init_worker,
        initargs=(
            lm_path,
            config.text_encoder.get("alphabet"),
            sp_model_path,
            topk,
            beam_width,
        ),
    ) as pool:
        for combo_id, cer_errors, wer_errors in tqdm(
            pool.imap_unordered(_decode_chunk, tasks),
            total=len(tasks),
            desc="Decoding",
        ):
            combo_cer[combo_id].extend(cer_errors)
            combo_wer[combo_id].extend(wer_errors)

    results = {}
    for part in partitions:
        results[part] = {
            "Argmax": argmax_results[part],
            "BeamSearch": {
                "cer": sum(combo_cer[f"{part}:BeamSearch"])
                / len(combo_cer[f"{part}:BeamSearch"]),
                "wer": sum(combo_wer[f"{part}:BeamSearch"])
                / len(combo_wer[f"{part}:BeamSearch"]),
            },
            "BeamSearchLM": {
                "cer": sum(combo_cer[f"{part}:BeamSearchLM"])
                / len(combo_cer[f"{part}:BeamSearchLM"]),
                "wer": sum(combo_wer[f"{part}:BeamSearchLM"])
                / len(combo_wer[f"{part}:BeamSearchLM"]),
            },
        }

    print(f"\nCheckpoint: {config.checkpoint_path}")
    print(f"LM: alpha={alpha}, beta={beta}, beam_width={beam_width}\n")
    print(f"{'partition':10s} {'variant':14s} {'CER':>8s} {'WER':>8s}")
    for part in partitions:
        for variant in ["Argmax", "BeamSearch", "BeamSearchLM"]:
            r = results[part][variant]
            print(f"{part:10s} {variant:14s} {r['cer']:8.4f} {r['wer']:8.4f}")

    with open(config.lm_eval.results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to '{config.lm_eval.results_path}'")


if __name__ == "__main__":
    main()
