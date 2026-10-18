_lm_cache = {}


def load_kenlm_model(lm_path):
    import kenlm

    if lm_path not in _lm_cache:
        _lm_cache[lm_path] = kenlm.Model(lm_path)
    return _lm_cache[lm_path]
