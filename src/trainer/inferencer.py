import logging
from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.metrics.decode_pool import _decode_one, get_pool
from src.metrics.tracker import MetricTracker
from src.text_encoder.lm.lm_utils import load_kenlm_model
from src.trainer.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        text_encoder,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            text_encoder (CTCTextEncoder): text encoder.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        self.text_encoder = text_encoder

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        self.save_beam_width = config.inferencer.get("save_beam_width", 10)
        lm_cfg = config.metrics.get("lm") if "metrics" in config else None
        if lm_cfg is not None and lm_cfg.get("path") is not None:
            self.lm_path = lm_cfg.path
            self.lm_alpha = lm_cfg.get("alpha", 0.5)
            self.lm_beta = lm_cfg.get("beta", 1.0)
        else:
            self.lm_path = None
            self.lm_alpha = 0.0
            self.lm_beta = 0.0
        self._decode_pool = get_pool(
            alphabet=text_encoder.alphabet,
            sp_model_path=getattr(text_encoder, "sp_model_path", None),
            topk=getattr(text_encoder, "beam_search_topk", 45),
        )

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(
            spectrogram=batch["spectrogram"],
            spectrogram_length=batch["spectrogram_length"],
        )
        batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        if self.save_path is not None:
            log_probs = batch["log_probs"].detach().cpu().numpy()
            log_probs_length = batch["log_probs_length"].detach().cpu().numpy()
            audio_paths = batch["audio_path"]

            out_dir = self.save_path / part
            out_dir.mkdir(exist_ok=True, parents=True)

            tasks = [
                (
                    log_probs[i, : int(log_probs_length[i])].copy(),
                    self.save_beam_width,
                    self.lm_path,
                    self.lm_alpha,
                    self.lm_beta,
                )
                for i in range(log_probs.shape[0])
            ]
            try:
                pred_texts = self._decode_pool.map(_decode_one, tasks)
            except Exception as e:
                logger.warning(
                    f"decode pool failed ({e!r}), falling back to serial "
                    "decoding for this batch"
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

            for i, pred_text in enumerate(pred_texts):
                utterance_id = Path(audio_paths[i]).stem
                (out_dir / f"{utterance_id}.txt").write_text(pred_text)

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result()
