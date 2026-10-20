"""
Sets up a shared pool of worker processes that decode beam search
predictions in parallel instead of one at a time
"""

import atexit
import logging
import multiprocessing as mp

from src.text_encoder.lm.lm_utils import load_kenlm_model

logger = logging.getLogger(__name__)

_pool = None
_pool_n_workers = None
_worker_encoder = None


def _init_worker(alphabet, sp_model_path, topk):
    """
    Builds a text encoder inside a worker process the first time that
    worker is used, so every worker can decode on its own

    Args:
        alphabet (list[str] | None): character alphabet to use, ignored
            if sp_model_path is given
        sp_model_path (str | None): path to a trained SentencePiece
            model, switches the encoder to BPE mode if given
        topk (int): how many candidates beam search considers per frame
    """
    global _worker_encoder
    from src.text_encoder.ctc_text_encoder import CTCTextEncoder

    _worker_encoder = CTCTextEncoder(
        alphabet=alphabet, sp_model_path=sp_model_path, topk=topk
    )


def _decode_one(task):
    """
    Decodes a single utterance inside a worker process, loading the
    language model for that path if one was given

    Args:
        task (tuple): a (log_probs, beam_width, lm_path, alpha, beta)
            tuple describing one utterance to decode.
    """
    log_probs, beam_width, lm_path, alpha, beta = task
    lm_model = load_kenlm_model(lm_path) if lm_path is not None else None
    return _worker_encoder.ctc_beam_search(
        log_probs, beam_width, lm_model=lm_model, alpha=alpha, beta=beta
    )


def default_n_workers():
    """
    Picks a reasonable number of worker processes
    """
    return max(1, mp.cpu_count() // 2)


def get_pool(n_workers=None, alphabet=None, sp_model_path=None, topk=45):
    """
    creating shared worker pool and reusing it after that

    Args:
        n_workers (int | None): how many worker processes to start, picks
            a sensible default if None.
        alphabet (list[str] | None): character alphabet for the workers'
            text encoders, ignored in BPE mode.
        sp_model_path (str | None): path to a trained SentencePiece
            model, switches the workers' encoders to BPE mode if given.
        topk (int): how many candidates beam search considers per frame.
    """
    global _pool, _pool_n_workers

    if n_workers is None:
        n_workers = default_n_workers()
    n_workers = max(1, int(n_workers))

    if _pool is not None:
        return _pool

    ctx = mp.get_context("spawn")
    _pool = ctx.Pool(
        n_workers, initializer=_init_worker, initargs=(alphabet, sp_model_path, topk)
    )
    _pool_n_workers = n_workers
    atexit.register(shutdown_pool)
    logger.info(f"Decode pool started with {n_workers} worker processes")
    return _pool


def shutdown_pool():
    """
    Shuts down the shared worker pool so a later call can start a fresh one
    """
    global _pool
    if _pool is not None:
        try:
            _pool.terminate()
            _pool.join()
        except Exception:
            pass
        _pool = None
