"""
Searches over language model weighting values to find the pair that
gives the lowest word error rate on the validation set
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

    Args:
        config (DictConfig): Hydra config with the dataset, model, and
            lm_tuning.cache_path settings
        device (str): device to run the acoustic model
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
    """
    Prepares one worker process with the cached samples, the language
    model, and a text encoder, so it can evaluate grid search points on
    its own

    Args:
        samples (list[dict]): cached (log_probs, text) pairs for the
            validation set
        lm_path (str): path to the KenLM model file to score hypotheses
            with
        alphabet (list[str] | None): character alphabet to use, ignored
            if sp_model_path is given
        sp_model_path (str | None): path to a trained SentencePiece
            model, switches the encoder to BPE mode if given
        topk (int): how many candidates beam search considers per frame
        beam_width (int): how many candidate hypotheses to keep at each
            time step
    """
    from src.text_encoder.ctc_text_encoder import CTCTextEncoder

    _WORKER["samples"] = samples
    _WORKER["lm"] = load_kenlm_model(lm_path)
    _WORKER["encoder"] = CTCTextEncoder(
        alphabet=alphabet, sp_model_path=sp_model_path, topk=topk
    )
    _WORKER["beam_width"] = beam_width


def _evaluate_combo(combo):
    """
    Decodes every cached validation sample with one alpha and beta
    setting and reports the resulting metrics

    Args:
        combo (tuple[float, float]): an (alpha, beta) pair to evaluate
    """
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
    """
    Runs the full search across every alpha and beta setting and prints
    the best found

    Args:
        config (DictConfig): Hydra config with the lm_tuning, metrics,
            and text_encoder settings for the grid search
    """
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
