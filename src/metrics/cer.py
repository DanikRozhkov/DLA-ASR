from src.metrics.base_metric import BaseDecodeMetric
from src.metrics.utils import calc_cer


class ArgmaxCERMetric(BaseDecodeMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(text_encoder, *args, calc_fn=calc_cer, **kwargs)

    def decode(self, log_probs_i):
        inds = log_probs_i.argmax(dim=-1).cpu().numpy()
        return self.text_encoder.ctc_decode(inds)


class BeamSearchCERMetric(BaseDecodeMetric):
    def __init__(self, text_encoder, beam_width=10, *args, **kwargs):
        super().__init__(text_encoder, *args, calc_fn=calc_cer, **kwargs)
        self.beam_width = beam_width

    def decode(self, log_probs_i):
        return self.text_encoder.ctc_beam_search(
            log_probs_i.cpu().numpy(), self.beam_width
        )
