"""
Character error rate metrics for argmax decoding and for beam search
decoding + optional LM
"""

from src.metrics.base_metric import BaseDecodeMetric, BaseParallelBeamSearchMetric
from src.metrics.utils import calc_cer


class ArgmaxCERMetric(BaseDecodeMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(text_encoder, *args, calc_fn=calc_cer, **kwargs)

    def decode(self, log_probs_i):
        """
        Picks the most likely token at every time step and turns the
        result into text

        Args:
            log_probs_i (Tensor): per-frame log-probabilities for one
                utterance
        """
        inds = log_probs_i.argmax(dim=-1).cpu().numpy()
        return self.text_encoder.ctc_decode(inds)


class BeamSearchCERMetric(BaseParallelBeamSearchMetric):
    def __init__(self, text_encoder, beam_width=10, n_workers=None, *args, **kwargs):
        """
        CER metric that decodes with beam search instead of argmax

        Args:
            text_encoder: encoder used to normalize text and decode
                predictions
            beam_width (int): how many candidate hypotheses to keep at
                each time step
            n_workers (int | None): how many worker processes the shared
                decode pool should use, picks kinda sensible default if None
        """
        super().__init__(
            text_encoder,
            calc_cer,
            beam_width=beam_width,
            n_workers=n_workers,
            *args,
            **kwargs,
        )


class BeamSearchLMCERMetric(BaseParallelBeamSearchMetric):
    def __init__(
        self,
        text_encoder,
        lm_path,
        beam_width=10,
        alpha=0.5,
        beta=1.0,
        n_workers=None,
        *args,
        **kwargs,
    ):
        """
        CER metric that decodes with beam search and language model
        fusion

        Args:
            text_encoder: encoder used to normalize text and decode
                predictions
            lm_path (str): path to a KenLM model file used to rescore
                hypotheses
            beam_width (int): how many candidate hypotheses to keep at
                each time step
            alpha (float): how much weight to give the language model's
                score relative to the acoustic model
            beta (float): bonus added per word, to counter the language
                model's bias toward shorter hypotheses
            n_workers (int | None): how many worker processes the shared
                decode pool should use, picks kinda sensible default if None
        """
        super().__init__(
            text_encoder,
            calc_cer,
            beam_width=beam_width,
            lm_path=lm_path,
            alpha=alpha,
            beta=beta,
            n_workers=n_workers,
            *args,
            **kwargs,
        )
