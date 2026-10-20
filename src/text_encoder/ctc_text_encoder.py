"""
Defines the text encoder with character and
BPE vocabularies with greedy and beam search decoding
"""

import math
import re
from collections import defaultdict
from string import ascii_lowercase

import numpy as np
import torch

NEG_INF = float("-inf")
WORD_BOUNDARY_MARKER = "▁"


def _log_add(a: float, b: float) -> float:
    """
    Numerically stable log sum exp

    Args:
        a (float): first log-probability
        b (float): second log-probability
    """
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, sp_model_path=None, topk=45, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """
        self.sp_model_path = sp_model_path
        self.is_bpe = sp_model_path is not None

        if self.is_bpe:
            import sentencepiece as spm

            self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)
            self.alphabet = None
            self.vocab = [self.EMPTY_TOK] + [
                self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())
            ]
        else:
            if alphabet is None:
                alphabet = list(ascii_lowercase + " ")
            self.alphabet = alphabet
            self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.beam_search_topk = topk
        _REFERENCE_VOCAB_SIZE = 28
        self._beam_search_cache = {}
        self._beam_search_cache_maxsize = max(
            256, int(8192 * _REFERENCE_VOCAB_SIZE / max(len(self.vocab), 1))
        )

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        if self.is_bpe:
            ids = [i + 1 for i in self.sp.encode_as_ids(text)]
            return torch.Tensor(ids).unsqueeze(0)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def _render(self, ids) -> str:
        """
        Turns a sequence of token ids back into readable text, using the
        SentencePiece model in BPE mode or plain character lookup
        otherwise

        Args:
            ids (list[int]): non-blank token indices to render as text
        """
        if self.is_bpe:
            return self.sp.decode_ids([int(i) - 1 for i in ids])
        return "".join(self.ind2char[int(i)] for i in ids)

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        if self.is_bpe:
            non_blank = [int(i) for i in inds if int(i) != 0]
            return self._render(non_blank).strip()
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        """
        Collapses repeated tokens and removes blanks from a raw
        prediction sequence, then renders what is left as text

        Args:
            inds (list[int]): raw per-frame predicted token indices,
                straight from argmax, before any collapsing
        """
        res = []
        counter = 0
        while counter < len(inds):
            while counter < len(inds) - 1 and inds[counter] == inds[counter + 1]:
                counter += 1
            res.append(inds[counter])
            counter += 1
        non_blank = [int(i) for i in res if int(i) != 0]
        return self._render(non_blank)

    def ctc_beam_search(
        self, log_probs, beam_width, lm_model=None, alpha=0.5, beta=1.0
    ) -> str:
        """
        CTC prefix beam search decoding, with optional LM fusion

        Args:
            log_probs (ndarray): per-frame log-probabilities for one
                utterance, shape (timestamps, vocab_size)
            beam_width (int): how many candidate hypotheses to keep at
                each time step
            lm_model: optional language model used to rescore hypotheses,
                left out entirely if None
            alpha (float): how much weight to give the language model's
                score relative to the acoustic model
            beta (float): bonus added per word, to counter the language
                model's bias toward shorter hypotheses
        Returns:
            best_prefix (str): decoded best hypothesis
        """
        cache_key = (
            log_probs.tobytes(),
            log_probs.shape,
            beam_width,
            id(lm_model) if lm_model is not None else None,
            alpha,
            beta,
        )
        cached = self._beam_search_cache.get(cache_key)
        if cached is not None:
            return cached

        def lm_score_for(ids):
            if lm_model is None:
                return 0.0
            text = self._render(ids).strip()
            if not text:
                return 0.0
            n_words = len(text.split(" "))
            return alpha * lm_model.score(text, bos=True, eos=False) + beta * n_words

        def is_word_boundary(prefix_ids, new_id):
            piece = self.ind2char[new_id]
            if self.is_bpe:
                return len(prefix_ids) > 0 and piece.startswith(WORD_BOUNDARY_MARKER)
            return piece == " "

        T, V = log_probs.shape
        blank_idx = self.char2ind[self.EMPTY_TOK]
        k = min(self.beam_search_topk, V)
        beam = {(): (0.0, NEG_INF, 0.0)}

        for t in range(T):
            frame = log_probs[t]
            if k >= V:
                top_idx = range(V)
            else:
                top_idx = np.argpartition(frame, -k)[-k:]
                if blank_idx not in top_idx:
                    top_idx = np.append(top_idx, blank_idx)

            new_beam = defaultdict(lambda: (NEG_INF, NEG_INF, 0.0))
            for prefix, (p_b, p_nb, lm_score) in beam.items():
                last_id = prefix[-1] if prefix else None
                p_total = _log_add(p_b, p_nb)
                for c_idx in top_idx:
                    c_idx = int(c_idx)
                    log_p_c = float(frame[c_idx])
                    if c_idx == blank_idx:
                        old_b, old_nb, _ = new_beam[prefix]
                        new_beam[prefix] = (
                            _log_add(old_b, p_total + log_p_c),
                            old_nb,
                            lm_score,
                        )
                    elif c_idx == last_id:
                        extended = prefix + (c_idx,)
                        old_b, old_nb, _ = new_beam[extended]
                        new_beam[extended] = (
                            old_b,
                            _log_add(old_nb, p_b + log_p_c),
                            lm_score,
                        )
                        old_b2, old_nb2, _ = new_beam[prefix]
                        new_beam[prefix] = (
                            old_b2,
                            _log_add(old_nb2, p_nb + log_p_c),
                            lm_score,
                        )
                    else:
                        new_prefix = prefix + (c_idx,)
                        if lm_model is not None and is_word_boundary(prefix, c_idx):
                            new_lm_score = lm_score_for(prefix)
                        else:
                            new_lm_score = lm_score
                        old_b, old_nb, _ = new_beam[new_prefix]
                        new_beam[new_prefix] = (
                            old_b,
                            _log_add(old_nb, p_total + log_p_c),
                            new_lm_score,
                        )
            beam = dict(
                sorted(
                    new_beam.items(),
                    key=lambda item: _log_add(item[1][0], item[1][1]) + item[1][2],
                    reverse=True,
                )[:beam_width]
            )

        def final_total(item):
            prefix, (p_b, p_nb, lm_score) = item
            base = _log_add(p_b, p_nb)
            if lm_model is None:
                return base
            return base + lm_score_for(prefix)

        best_ids = max(beam.items(), key=final_total)[0]
        best_prefix = self._render(best_ids)

        if len(self._beam_search_cache) >= self._beam_search_cache_maxsize:
            self._beam_search_cache.clear()
        self._beam_search_cache[cache_key] = best_prefix

        return best_prefix

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
