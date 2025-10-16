# standard lib

# third party
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

import common.audio

# collection of colors for visualizations
COLORS = [
    ("lightskyblue", "steelblue"),
    ("palegreen", "mediumseagreen"),
    ("plum", "mediumorchid"),
    ("tomato", "firebrick"),
    ("navajowhite", "burlywood"),
    ("paleturquoise", "teal"),
]


def plot_frames(f: torch.Tensor, sr, title="frames", ax=None, frame_offset=0) -> None:
    if f.dim() > 1:
        raise Exception("Expected 1 dimensional tensor")

    n_frames = len(f)

    time_axis = (torch.arange(frame_offset, n_frames + frame_offset) / (sr / 1000))

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax.plot(time_axis, f, color='#394f4b', lw=1)
    ax.set_title(title)


def plot_spectrogram(s, title="Spectrogram", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax.imshow(librosa.power_to_db(s), origin="lower", aspect="auto", interpolation="nearest")
    ax.set_title(title)


def plot_mfcc(m: torch.Tensor, title="MFCC", ax=None):
    m = m.numpy()

    if ax is None:
        _, ax = plt.subplots(1, 1)

    # ax.imshow(librosa.power_to_db(m[0]), origin="lower", aspect="auto", interpolation="nearest")
    # ax.imshow(m[0], origin="lower", aspect="auto", interpolation="nearest", cmap="viridis")
    ax.imshow(m[0], origin="lower", aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energies")
    ax.set_title(title)


def plot_hist_from_df(
        ax,
        pd_series,
        series_label: str,
        colors: tuple[str, str, float],
        bin_size: float,
        boundaries: tuple[float, float] | None,
        use_density: bool = False,
        x_label: str = "x axis",
        y_label: str = None,
        title: str = None
):
    ax.set_xlabel(x_label)

    if y_label:
        ax.set_ylabel(y_label)

    if title:
        ax.set_title(title)

    boundaries is boundaries or (
        math.floor(pd_series.min() / bin_size) * bin_size, math.ceil(pd_series.max() / bin_size) * bin_size)

    bins = np.arange(boundaries[0], boundaries[1] + 10e-3, bin_size)

    ax.hist(pd_series, color=colors[0], edgecolor=colors[1], alpha=colors[2],
            label=series_label,
            bins=bins,
            density=use_density)
