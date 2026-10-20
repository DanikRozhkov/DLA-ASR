"""
A standalone script that logs a handful of audio examples before and
after every augmentation
"""

import logging

import hydra
import soundfile
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.logger.utils import plot_spectrogram, plot_waveform
from src.utils.io_utils import ROOT_PATH

logger = logging.getLogger(__name__)


def unique_by_type(transform_list):
    """
    Keeps only the first example of each distinct augmentation so the
    demo does not repeat the same kind of transform twice

    Args:
        transform_list (list): augmentations to filter down to one per
            type
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
    Turns an augmentations probability up to 1 so it always applies for the demo

    Args:
        transform: augmentation to force on, optionally wrapped in a
            RandomApply-style container
    """
    inner = getattr(transform, "_aug", transform)
    if hasattr(inner, "p"):
        inner.p = 1.0
    return transform


@hydra.main(
    version_base=None, config_path="src/configs", config_name="show_augumentation"
)
def main(config):
    """
    Runs every augmentation on a sample from dev-clean and logs the
    results to Comet so they can be checked by ear and by eye

    Args:
        config (DictConfig): Hydra config with the transforms and writer
            settings
    """
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
        clean_audio_np, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
        clean_audio = torch.from_numpy(clean_audio_np.T).contiguous()[0:1, :]
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

    logger.info("Done")


if __name__ == "__main__":
    main()
