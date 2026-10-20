"""
Small plotting helpers that turn spectrograms and waveforms into images
for logging
"""

import io

import matplotlib.pyplot as plt
import PIL
import torch
from torchvision.transforms import ToTensor

plt.switch_backend("agg")  # fix RuntimeError: main thread is not in main loop


def plot_images(imgs, config):
    """
    Combine several images into one figure.

    Args:
        imgs (Tensor): array of images (B X C x H x W).
        config (DictConfig): hydra experiment config.
    Returns:
        image (Tensor): a single figure with imgs plotted side-to-side.
    """
    # name of each img in the array
    names = config.writer.names
    # figure size
    figsize = config.writer.figsize
    fig, axes = plt.subplots(1, len(names), figsize=figsize)
    for i in range(len(names)):
        # channels must be in the last dim
        img = imgs[i].permute(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_title(names[i])
        axes[i].axis("off")  # we do not need axis
    # To create a tensor from matplotlib,
    # we need a buffer to save the figure
    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    # convert buffer to Tensor
    image = ToTensor()(PIL.Image.open(buf))

    plt.close()

    return image


def plot_spectrogram(spectrogram, name=None):
    """
    Plot spectrogram

    Args:
        spectrogram (Tensor): spectrogram tensor.
        name (None | str): optional name.
    Returns:
        image (Image): image of the spectrogram
    """
    plt.figure(figsize=(20, 5))
    plt.pcolormesh(spectrogram)
    plt.title(name)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # convert buffer to Tensor
    image = ToTensor()(PIL.Image.open(buf))

    plt.close()

    return image


def plot_waveform(audio, sample_rate, name=None):
    """
    Draws a waveform over time and returns it as an
    image that is ready for logging

    Args:
        audio (Tensor): waveform samples, only the first channel is
            used if multi-channel
        sample_rate (int): sample rate of the audio, used to build the
            time axis
        name (str | None): optional title for the plot
    """
    audio = audio.detach().cpu()
    if audio.dim() > 1:
        audio = audio[0]
    time_axis = torch.arange(audio.shape[0]) / sample_rate

    plt.figure(figsize=(20, 5))
    plt.plot(time_axis, audio.numpy())
    plt.xlabel("Time - seconds")
    plt.grid()
    plt.title(name)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    image = ToTensor()(PIL.Image.open(buf))

    plt.close()

    return image
