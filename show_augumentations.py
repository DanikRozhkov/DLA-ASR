import logging

import hydra
import torchaudio
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.logger.utils import plot_spectrogram, plot_waveform
from src.utils.io_utils import ROOT_PATH

logger = logging.getLogger(__name__)


def unique_by_type(transform_list):
    """
    Keep only the first instance of each distinct transform class
    """
    seen_types = set()
    unique = []
    for t in transform_list:
        if type(t) not in seen_types:
            seen_types.add(type(t))
            unique.append(t)
    return unique


def force_apply(transform):
    """
    Force transform to always fire for the demo
    """
    inner = getattr(transform, "_aug", transform)
    if hasattr(inner, "p"):
        inner.p = 1.0
    return transform


@hydra.main(
    version_base=None, config_path="src/configs", config_name="show_augumentation"
)
def main(config):
    project_config = OmegaConf.to_container(config)
    writer = instantiate(config.writer, logger, project_config)
    get_spectrogram = instantiate(
        config.transforms.instance_transforms.inference.get_spectrogram
    )
    full_audio_pipeline = instantiate(config.transforms.instance_transforms.train.audio)
    full_spec_pipeline = instantiate(
        config.transforms.instance_transforms.train.spectrogram
    )
    demo_audio_augs = unique_by_type(
        instantiate(config.transforms.instance_transforms.train.audio).transforms
    )
    demo_spec_augs = unique_by_type(
        instantiate(config.transforms.instance_transforms.train.spectrogram).transforms
    )
    for aug in demo_audio_augs:
        force_apply(aug)

    audio_dir = ROOT_PATH / config.audio_dir
    audio_paths = sorted(audio_dir.rglob(config.audio_glob))[: config.n_examples]

    writer.set_step(0, mode="augmentation_demo")
    for i, path in enumerate(audio_paths):
        clean_audio, sr = torchaudio.load(str(path))
        clean_spec = get_spectrogram(clean_audio)

        writer.add_audio(f"sample_{i}/clean_audio", clean_audio, sample_rate=sr)
        writer.add_image(f"sample_{i}/clean_waveform", plot_waveform(clean_audio, sr))
        writer.add_image(
            f"sample_{i}/clean_spectrogram", plot_spectrogram(clean_spec[0])
        )

        for aug in demo_audio_augs:
            name = type(aug).__name__
            aug_audio = aug(clean_audio)
            aug_spec = get_spectrogram(aug_audio)
            writer.add_audio(f"sample_{i}/{name}_audio", aug_audio, sample_rate=sr)
            writer.add_image(
                f"sample_{i}/{name}_waveform", plot_waveform(aug_audio, sr)
            )
            writer.add_image(
                f"sample_{i}/{name}_spectrogram", plot_spectrogram(aug_spec[0])
            )

        for aug in demo_spec_augs:
            name = type(aug).__name__
            aug_spec = aug(clean_spec)
            writer.add_image(
                f"sample_{i}/{name}_spectrogram", plot_spectrogram(aug_spec[0])
            )

        combined_audio = full_audio_pipeline(clean_audio)
        combined_spec = full_spec_pipeline(get_spectrogram(combined_audio))
        writer.add_audio(f"sample_{i}/combined_audio", combined_audio, sample_rate=sr)
        writer.add_image(
            f"sample_{i}/combined_waveform", plot_waveform(combined_audio, sr)
        )
        writer.add_image(
            f"sample_{i}/combined_spectrogram", plot_spectrogram(combined_spec[0])
        )

        logger.info(f"Logged sample {i}: {path.name}")

    logger.info("Done - check the W&B run for before/after audio and spectrograms")


if __name__ == "__main__":
    main()
