import atexit
import logging
import multiprocessing as mp

from src.text_encoder.lm.lm_utils import load_kenlm_model

logger = logging.getLogger(__name__)

_pool = None
_pool_n_workers = None
_worker_encoder = None


def _init_worker(alphabet):
    global _worker_encoder
    from src.text_encoder.ctc_text_encoder import CTCTextEncoder

    _worker_encoder = CTCTextEncoder(alphabet=alphabet)


def _decode_one(task):
    log_probs, beam_width, lm_path, alpha, beta = task
    lm_model = load_kenlm_model(lm_path) if lm_path is not None else None
    return _worker_encoder.ctc_beam_search(
        log_probs, beam_width, lm_model=lm_model, alpha=alpha, beta=beta
    )


def default_n_workers():
    return max(1, mp.cpu_count() // 2)


def get_pool(n_workers=None, alphabet=None):
    """
    Return the shared decode pool, creating it on first call. Subsequent
    calls ignore n_workers/alphabet and return the same pool -- metrics
    are all constructed once at Trainer startup with the same text
    encoder, so this is a non-issue in practice; it's documented here
    because it's a real constraint of the singleton design.
    """
    global _pool, _pool_n_workers

    if n_workers is None:
        n_workers = default_n_workers()
    n_workers = max(1, int(n_workers))

    if _pool is not None:
        return _pool

    ctx = mp.get_context("spawn")
    _pool = ctx.Pool(n_workers, initializer=_init_worker, initargs=(alphabet,))
    _pool_n_workers = n_workers
    atexit.register(shutdown_pool)
    logger.info(f"Decode pool started with {n_workers} worker processes")
    return _pool


def shutdown_pool():
    global _pool
    if _pool is not None:
        try:
            _pool.terminate()
            _pool.join()
        except Exception:
            pass
        _pool = None
