from abc import abstractmethod


class BaseMetric:
    """
    Base class for all metrics
    """

    def __init__(self, name=None, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        self.name = name if name is not None else type(self).__name__

    @abstractmethod
    def __call__(self, **batch):
        """
        Defines metric calculation logic for a given batch.
        Can use external functions (like TorchMetrics) or custom ones.
        """
        raise NotImplementedError()


class BaseDecodeMetric(BaseMetric):
    """
    Shared logic for metrics that decode model log_probs into text and
    compare it to the target text via some error-rate function (CER/WER)
    See src/metrics/wer.py and src/metrics/cer.py
    """

    def __init__(self, text_encoder, calc_fn, decode_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.calc_fn = calc_fn
        self.decode_fn = decode_fn

    def __call__(self, log_probs, log_probs_length, text, **kwargs):
        lengths = log_probs_length.detach().numpy()
        errors = []
        for log_prob_vec, length, target_text in zip(log_probs, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.decode_fn(self.text_encoder, log_prob_vec[:length])
            errors.append(self.calc_fn(target_text, pred_text))
        return sum(errors) / len(errors)

    def decode(self, log_probs_i):
        """
        Decode a single utterance's log_probs into text.
        Will be implemented by subclasses
        """
        raise NotImplementedError()


def argmax_decode(text_encoder, log_probs_i):
    inds = log_probs_i.argmax(dim=-1).cpu().numpy()
    return text_encoder.ctc_decode(inds)


def beam_search_decode(text_encoder, log_probs_i, beam_width):
    return text_encoder.ctc_beam_search(log_probs_i.cpu().numpy(), beam_width)
