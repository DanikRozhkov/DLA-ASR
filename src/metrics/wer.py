from src.metrics.base_metric import BaseDecodeMetric, BaseParallelBeamSearchMetric
from src.metrics.utils import calc_wer


class ArgmaxWERMetric(BaseDecodeMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(text_encoder, *args, calc_fn=calc_wer, **kwargs)

    def decode(self, log_probs_i):
        inds = log_probs_i.argmax(dim=-1).cpu().numpy()
        return self.text_encoder.ctc_decode(inds)


class BeamSearchWERMetric(BaseParallelBeamSearchMetric):
    def __init__(self, text_encoder, beam_width=10, n_workers=None, *args, **kwargs):
        super().__init__(
            text_encoder,
            calc_wer,
            beam_width=beam_width,
            n_workers=n_workers,
            *args,
            **kwargs,
        )


class BeamSearchLMWERMetric(BaseParallelBeamSearchMetric):
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
        super().__init__(
            text_encoder,
            calc_wer,
            beam_width=beam_width,
            lm_path=lm_path,
            alpha=alpha,
            beta=beta,
            n_workers=n_workers,
            *args,
            **kwargs,
        )
