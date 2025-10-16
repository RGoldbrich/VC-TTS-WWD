import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def hertz_to_mel(freq):
    return 2595 * np.log10(1 + freq / 700)


def mel_to_hertz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


# Example Usage
n_filters = 12
sr = 16000
f_min = 0
f_max = sr // 2  # Nyquist frequency

f_min_mel = 0
f_max_mel = hertz_to_mel(f_max)

# linear scaled in the mel scale
filter_peaks_mel = np.linspace(f_min_mel, f_max_mel, n_filters)
mel_filter_spacing = (f_max_mel / (n_filters - 1))

# filter peaks in the hertz scale
filter_peaks_hertz = mel_to_hertz(filter_peaks_mel)

triangle_geometry = []

for peak_hz in filter_peaks_hertz:
    print(peak_hz)

    filter_width_mel = mel_filter_spacing

    lf = mel_to_hertz(hertz_to_mel(peak_hz) - filter_width_mel)
    rf = mel_to_hertz(hertz_to_mel(peak_hz) + filter_width_mel)

    # height of triangle
    # length(base) * height / 2 = area
    # 2 * area / length(base) = height

    height = 2 * (1) / (rf - lf)

    triangle_geometry.append((
        (lf, rf, peak_hz, lf),
        (0, 0, height, 0)
    ))

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 20

fig = plt.figure(figsize=(12, 5))
axs = fig.add_subplot()
# plt.figure(figsize=(12, 5))

axs.set_xlim(-500, f_max * 1.3)
axs.set_ylim(0, 0.007)

print(tuple(np.random.random(size=3) * 256))

for tr in triangle_geometry:
    axs.plot(tr[0], tr[1], color=tuple(np.random.random(size=3)))

axs.set_xlabel("Frequency (Hz)")
axs.set_ylabel("Filter Response")
fig.tight_layout()
plt.show()

fig.savefig("MelScaleFilters.svg", format="svg")
