import logging
from abc import abstractmethod

from src.metrics.decode_pool import _decode_one, get_pool
from src.text_encoder.lm.lm_utils import load_kenlm_model

logger = logging.getLogger(__name__)


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

    def __init__(self, text_encoder, calc_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.calc_fn = calc_fn

    def __call__(self, log_probs, log_probs_length, text, **kwargs):
        lengths = log_probs_length.detach().cpu().numpy()
        errors = []
        for log_prob_vec, length, target_text in zip(log_probs, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.decode(log_prob_vec[:length])
            errors.append(self.calc_fn(target_text, pred_text))
        return sum(errors) / len(errors)

    def decode(self, log_probs_i):
        """
        Decode a single utterance's log_probs into text.
        Will be implemented by subclasses
        """
        raise NotImplementedError()


class BaseParallelBeamSearchMetric(BaseDecodeMetric):
    def __init__(
        self,
        text_encoder,
        calc_fn,
        beam_width=10,
        lm_path=None,
        alpha=0.0,
        beta=0.0,
        n_workers=None,
        *args,
        **kwargs,
    ):
        super().__init__(text_encoder, calc_fn, *args, **kwargs)
        self.beam_width = beam_width
        self.lm_path = lm_path
        self.alpha = alpha
        self.beta = beta
        self.n_workers = n_workers
        self._pool = get_pool(n_workers=n_workers, alphabet=text_encoder.alphabet)

    def __call__(self, log_probs, log_probs_length, text, **kwargs):
        if log_probs.shape[0] == 0:
            return 0.0

        log_probs_np = log_probs.detach().cpu().numpy()
        lengths = log_probs_length.detach().cpu().numpy()

        tasks = [
            (
                log_probs_np[i, : int(lengths[i])].copy(),
                self.beam_width,
                self.lm_path,
                self.alpha,
                self.beta,
            )
            for i in range(log_probs_np.shape[0])
        ]

        try:
            pred_texts = self._pool.map(_decode_one, tasks)
        except Exception as e:
            logger.warning(
                f"{self.name}: decode pool failed ({e!r}), falling back to "
                "serial decoding for this batch"
            )
            lm_model = (
                load_kenlm_model(self.lm_path) if self.lm_path is not None else None
            )
            pred_texts = [
                self.text_encoder.ctc_beam_search(
                    lp, bw, lm_model=lm_model, alpha=a, beta=b
                )
                for lp, bw, _, a, b in tasks
            ]

        errors = [
            self.calc_fn(self.text_encoder.normalize_text(target_text), pred_text)
            for target_text, pred_text in zip(text, pred_texts)
        ]
        return sum(errors) / len(errors)
