import math
import re
from collections import defaultdict
from string import ascii_lowercase

import torch

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier

NEG_INF = float("-inf")


def _log_add(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        res = []
        counter = 0
        while counter < len(inds):
            while counter < len(inds) - 1 and inds[counter] == inds[counter + 1]:
                counter += 1
            res.append(inds[counter])
            counter += 1
        return "".join([self.ind2char[int(i)] if i != 0 else "" for i in res])

    def ctc_beam_search(self, log_probs, beam_width) -> str:
        """
        CTC prefix beam search decoding

        Args:
            log_probs (Tensor): log probabilities, shape (timestamps, vocab_size)
            beam_width (int): number of candidate prefixes keeps at each step
        Returns:
            best_prefix (str): decoded best hypothesis
        """
        T, V = log_probs.shape
        blank_idx = self.char2ind[self.EMPTY_TOK]
        beam = {"": (0.0, NEG_INF)}

        for t in range(T):
            frame = log_probs[t]
            new_beam = defaultdict(lambda: (NEG_INF, NEG_INF))
            for prefix, (p_b, p_nb) in beam.items():
                last_char = prefix[-1] if prefix else None
                p_total = _log_add(p_b, p_nb)
                for c_idx in range(V):
                    log_p_c = float(frame[c_idx])
                    c = self.ind2char[c_idx]
                    if c_idx == blank_idx:
                        old_b, old_nb = new_beam[prefix]
                        new_beam[prefix] = (_log_add(old_b, p_total + log_p_c), old_nb)
                    elif c == last_char:
                        old_b, old_nb = new_beam[prefix + c]
                        new_beam[prefix + c] = (old_b, _log_add(old_nb, p_b + log_p_c))
                        old_b2, old_nb2 = new_beam[prefix]
                        new_beam[prefix] = (old_b2, _log_add(old_nb2, p_nb + log_p_c))
                    else:
                        old_b, old_nb = new_beam[prefix + c]
                        new_beam[prefix + c] = (
                            old_b,
                            _log_add(old_nb, p_total + log_p_c),
                        )
            beam = dict(
                sorted(
                    new_beam.items(),
                    key=lambda item: _log_add(item[1][0], item[1][1]),
                    reverse=True,
                )[:beam_width]
            )

        best_prefix = max(
            beam.items(), key=lambda item: _log_add(item[1][0], item[1][1])
        )[0]
        return best_prefix

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
