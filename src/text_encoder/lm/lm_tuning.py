"""
Grid-search alpha/beta for LM shallow fusion on the validation set.

The acoustic model is run exactly once over the val set to produce
log_probs (the expensive, GPU-bound step, independent of alpha/beta).
Every (alpha, beta) combination is then evaluated purely on CPU by
replaying the cached log_probs through ctc_beam_search, grid points
split across worker processes since they are fully independent of
each other.
"""

import itertools
import json
import multiprocessing as mp
import os
import pickle
import warnings

import hydra
import torch
from tqdm import tqdm

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder.lm.lm_utils import load_kenlm_model
from src.utils.eval_utils import extract_log_probs
from src.utils.init_utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)


def extract_val_log_probs(config, device):
    """
    Run the acoustic model once over the val set and cache (log_probs,
    text) for every sample to disk. Skips recomputation on later runs
    if the cache file already exists.
    """
    cache_path = config.lm_tuning.cache_path
    if os.path.exists(cache_path):
        print(f"Loading cached val log_probs from '{cache_path}'")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    samples = extract_log_probs(config, "val", device)

    print(f"Caching {len(samples)} val samples -> '{cache_path}'")
    with open(cache_path, "wb") as f:
        pickle.dump(samples, f)
    return samples


_WORKER = {}


def _init_worker(samples, lm_path, alphabet, sp_model_path, topk, beam_width):
    from src.text_encoder.ctc_text_encoder import CTCTextEncoder

    _WORKER["samples"] = samples
    _WORKER["lm"] = load_kenlm_model(lm_path)
    _WORKER["encoder"] = CTCTextEncoder(
        alphabet=alphabet, sp_model_path=sp_model_path, topk=topk
    )
    _WORKER["beam_width"] = beam_width


def _evaluate_combo(combo):
    alpha, beta = combo
    samples = _WORKER["samples"]
    lm = _WORKER["lm"]
    encoder = _WORKER["encoder"]
    beam_width = _WORKER["beam_width"]

    cer_errors, wer_errors = [], []
    for s in samples:
        target = encoder.normalize_text(s["text"])
        pred = encoder.ctc_beam_search(
            s["log_probs"], beam_width, lm_model=lm, alpha=alpha, beta=beta
        )
        cer_errors.append(calc_cer(target, pred))
        wer_errors.append(calc_wer(target, pred))

    return {
        "alpha": alpha,
        "beta": beta,
        "cer": sum(cer_errors) / len(cer_errors),
        "wer": sum(wer_errors) / len(wer_errors),
    }


@hydra.main(version_base=None, config_path="../../configs", config_name="lm_tuning")
def main(config):
    set_random_seed(config.trainer.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.trainer.device != "auto":
        device = config.trainer.device

    samples = extract_val_log_probs(config, device)

    alphas = list(config.lm_tuning.alphas)
    betas = list(config.lm_tuning.betas)
    grid = list(itertools.product(alphas, betas))
    beam_width = config.lm_tuning.beam_width
    lm_path = config.metrics.lm.path
    sp_model_path = config.text_encoder.get("sp_model_path")
    topk = config.text_encoder.get("topk", 45)
    n_workers = config.lm_tuning.n_workers or mp.cpu_count()

    print(
        f"Grid-searching {len(grid)} (alpha, beta) combinations over "
        f"{len(samples)} val samples using {n_workers} worker processes"
    )

    results = []
    with mp.Pool(
        n_workers,
        initializer=_init_worker,
        initargs=(
            samples,
            lm_path,
            config.text_encoder.get("alphabet"),
            sp_model_path,
            topk,
            beam_width,
        ),
    ) as pool:
        for r in tqdm(
            pool.imap_unordered(_evaluate_combo, grid),
            total=len(grid),
            desc="Grid search",
        ):
            results.append(r)

    results.sort(key=lambda r: r["wer"])

    print("\n alpha   beta      CER       WER")
    for r in results:
        print(f"{r['alpha']:6.2f} {r['beta']:6.2f}   {r['cer']:.4f}   {r['wer']:.4f}")

    best = results[0]
    print(
        f"\nBest by WER: alpha={best['alpha']}, beta={best['beta']} "
        f"(CER={best['cer']:.4f}, WER={best['wer']:.4f})"
    )

    with open(config.lm_tuning.results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full grid results saved to '{config.lm_tuning.results_path}'")


if __name__ == "__main__":
    main()
