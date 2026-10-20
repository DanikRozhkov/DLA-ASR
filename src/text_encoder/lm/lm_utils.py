"""
A tiny cache so the same language model file is only loaded once per process
"""

_lm_cache = {}


def load_kenlm_model(lm_path):
    """
    Loads a KenLM language model from disk

    Args:
        lm_path (str): path to the KenLM model file on disk
    """
    import kenlm

    if lm_path not in _lm_cache:
        _lm_cache[lm_path] = kenlm.Model(lm_path)
    return _lm_cache[lm_path]
