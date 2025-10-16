# standard lib
import json
import math
import os

# third party
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import torch
import torchaudio
from matplotlib import gridspec
import matplotlib

# application
import common.visualizations as visuc
from alignment import clean_transcript
from common.audio import to_mono_drop_ch_dim

color_palette = {
    "HUM": "#8c9c5460",  # earthy, olive, moss
    "TTS+VC": "#b83064c0",  # red deep purple
    "TTS": "#305582c0",  # blue
    "VC": "#F9A602c0",  # yellow sunrise
}


def vis_inspect(source_dirs: list[str], series_labels: list[str] = None) -> None:
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2, fig)

    ax_duration = fig.add_subplot(gs[0, :])
    ax_mean_volume = fig.add_subplot(gs[1, 1])
    ax_max_volume = fig.add_subplot(gs[1, 0])

    dfs = [pd.read_csv(os.path.join(s, "info.csv")) for s in source_dirs]

    for idx, df in enumerate(dfs):
        df["series_label"] = series_labels[idx]

    df = pd.concat(dfs, ignore_index=True)
    print(df.columns)
    print(df.shape)

    filtered = df[df["duration_s"] < 0.5]
    print(filtered.shape)

    filtered = df[df["duration_s"] > 1.5]
    print(filtered.shape)

    sns.histplot(df, x="duration_s", ax=ax_duration, binrange=(0, 2), binwidth=0.025,
                 legend=False, color=color_palette["ORG"], alpha=0.75, stat="density")
    ax_duration.set_title("Duration")
    ax_duration.set_xlabel("Duration [s]")
    ax_duration.set_ylabel("Density")

    sns.histplot(df, x="vol_mean_db", ax=ax_mean_volume, binwidth=1,
                 legend=False, color=color_palette["ORG"], alpha=0.75, stat="density")
    ax_mean_volume.set_title("Mean Volume")
    ax_mean_volume.set_xlabel("Volume [db]")
    ax_mean_volume.set_ylabel("Density")

    sns.histplot(df, x="vol_max_db", ax=ax_max_volume, binwidth=1,
                 legend=False, color=color_palette["ORG"], alpha=0.75, stat="density")
    ax_max_volume.set_title("Max Volume")
    ax_max_volume.set_xlabel("Volume [db]")
    ax_max_volume.set_ylabel("Density")

    plt.tight_layout()
    plt.show()

    fig.savefig("Inspect_ORG.svg", format="svg")


def plot_hist(source_dirs: list[list[str]], series_labels: list[list[str]] = None) -> None:
    # fig, axs = plt.subplots(3, 2, figsize=(18, 18))
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    for idx, source_dir in enumerate(source_dirs):
        dfs = [pd.read_csv(os.path.join(s, "info.csv")) for s in source_dir]
        # ax_idx = (idx // 2, idx % 2)
        ax_idx = idx

        for i, df in enumerate(dfs):
            # df["series_label"] = series_labels[idx][i]

            color_key = series_labels[idx][i].split(" ")[0]

            sns.histplot(df, x="vol_max_db", ax=axs[ax_idx], legend=False, binwidth=1, binrange=(-50, 0),
                         color=color_palette[color_key], alpha=None,
                         stat="density")

        # df = pd.concat(dfs, ignore_index=True)

        # sns.kdeplot(df, x="vol_max_db", hue="series_label", ax=axs[ax_idx], fill=False, legend=False)
        # sns.histplot(df, x="vol_max_db", hue="series_label", ax=axs[ax_idx], legend=False, binwidth=1)

        # axs[ax_idx].set_title(series_labels[idx][0])
        axs[ax_idx].set_title("Before" if idx == 0 else "After")

        axs[ax_idx].set_xlabel("Volume [db]")
        # axs[ax_idx].set_xlabel("Duration [s]")
        axs[ax_idx].set_ylabel("Density")

        axs[ax_idx].set_xlim((-50, 5))
        # axs[ax_idx].set_xlim((-0.25, 2.25))

        axs[ax_idx].set_ylim((0, 0.25))
        # axs[ax_idx].set_ylim((0, 5))

        axs[ax_idx].legend(series_labels[idx], loc="upper left")
        # axs[ax_idx].legend(series_labels[idx], loc="upper right")

    plt.tight_layout()
    plt.show()

    fig.savefig("out.svg", format="svg")


def vis_feature(
        feature_path: str,
        label: str = "Log Mel Filterbank Energies"
) -> None:
    # fig = plt.figure(figsize=(8.4, 4.8))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot()

    feat = torch.load(feature_path)[0]
    print(feat[:, :, 120])
    visuc.plot_mfcc(feat, label, ax)
    plt.tight_layout()
    plt.show()

    fig.savefig("out.svg", format="svg")

    print(feat.shape)


def print_preprocessing_params(para: dict, skip_origin=False) -> None:
    if not skip_origin:
        print("Originating from", para['source_dir'])
        print(f"  and labeled: {para['labels']}")

    print("Processed:")
    if para['mode'] == "no_cut":
        print("  mode: no_cut with hop_length_ms", para['hop_length_ms'])
    else:
        print("  mode:", para['mode'])

    if para['src_content_length_ms']:
        print(f"  filtered to {para['src_content_length_ms']} using {para['vad_content_length_mode']}")

    if len(para['contamination_settings']) != 0:
        print(f"  contaminated using", para['contamination_settings'])

    if para['cdf_volume_transform']:
        print(f"  cdf: {para['cdf_volume_transform']}")

    if para['rir_likelihood'] > 0.0:
        print(f"  rir: {para['rir_likelihood']}")


def display_feature_origin(source: list[tuple[str, int, float]]) -> None:
    for s in source:
        with open(f"{s[0]}/feature_info.json", mode="r") as feat_f:
            feature_info = json.load(feat_f)

        # additionally load preprocessing info when this dataset is not directly derived from a source dataset
        if "out_preparation" in feature_info["preprocessing_params"]["source_dir"]:
            with open(feature_info["preprocessing_params"]["source_dir"] + "/preprocessing_info.json",
                      mode="r") as prep_f:
                preprocessing_info = json.load(prep_f)

            print_preprocessing_params(preprocessing_info["preprocessing_params"])

        print_preprocessing_params(feature_info["preprocessing_params"], True)

        # ensure sane cdf source
        assert "".join(feature_info["preprocessing_params"]["source_dir"].split("_")[-3]) in \
               feature_info["preprocessing_params"]["cdf_volume_transform"][0] or not \
                   feature_info["preprocessing_params"]["cdf_volume_transform"]

        # ensure sane cdf target
        assert "HotWord" in feature_info["preprocessing_params"]["cdf_volume_transform"][1] and feature_info[
            "class"] == "HotWord" or "HotWord" not in feature_info["preprocessing_params"]["cdf_volume_transform"][
                   1] and feature_info["class"] != "HotWord"

        if feature_info["is_training"]:
            print("TRAINING data with label:", feature_info["class"] == "HotWord")
        else:
            print("TEST data with label:", feature_info["class"] == "HotWord")

        print()


def show_wave(file: str) -> None:
    w, sr = torchaudio.load(file)

    w = to_mono_drop_ch_dim(w)

    visuc.plot_frames(w, sr)
    plt.show()


def vis_alignment_confidence(source_dirs: list[str]):
    fig, axs = plt.subplots(3, 2, figsize=(18, 18))
    # fig, axs = plt.subplots(1, 1, figsize=(12, 8))

    for idx, source_dir in enumerate(source_dirs):
        with open(source_dir + "/confidence.json") as f:
            alignment_json = json.load(f)

        flat_data = []
        for file_key, file_dict in alignment_json.items():
            for phrase_key, phrase_dict in file_dict.items():
                flat_row = {"filename": file_key, "phrase": phrase_key}
                flat_row.update(phrase_dict)
                flat_data.append(flat_row)

        df = pd.DataFrame(flat_data)
        print(df.columns)

        df["overall_confidence"] = df["phoneme_gen_mean_p2"] * df["filler_gen_mean_p-2"]

        # keys:
        # ['filename', 'phrase', 'phoneme_ari_mean', 'phoneme_gen_mean_p2', 'phoneme_gen_mean_p3', 'phoneme_gen_mean_p4',
        # 'filler_ari_mean', 'filler_gen_mean_p-1', 'filler_gen_mean_p-2', 'filler_gen_mean_p-3']

        phoneme_cols = ['phoneme_ari_mean', 'phoneme_gen_mean_p2', 'phoneme_gen_mean_p3', 'phoneme_gen_mean_p4']
        filler_cols = ['filler_ari_mean', 'filler_gen_mean_p-1', 'filler_gen_mean_p-2', 'filler_gen_mean_p-3']

        # for col in filler_cols:
        #     sns.kdeplot(df[col], label=col)

        ax_idx = (idx // 2, idx % 2)
        color_key = source_dir.split("/")[-1].split("_")[0]
        # color_key = "ORG"

        ax = axs[ax_idx]
        # ax = axs

        # sns.kdeplot(df["overall_confidence"], label="overall_confidence", fill=True, ax=axs[ax_idx])
        sns.histplot(df["overall_confidence"], ax=ax, color=color_palette[color_key],
                     stat="density", binrange=(0, 1), binwidth=0.02, alpha=0.75)
        ax.set_xlabel('Overall Confidence')
        ax.set_ylabel('Density')

        label = color_key + " - " + source_dir.split("/")[-1].split("_")[2]
        # label = "ORG - Snips"
        if label == "VC - Sonos":
            label = "VC - Snips"

        ax.set_title(label)

    plt.tight_layout()
    plt.show()
    fig.savefig("Conf_Dist.svg", format="svg")


def vis_grapheme_confidence(source_dirs: list[str]):
    # fig, axs = plt.subplots(3, 2, figsize=(18, 18))
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))

    for idx, source_dir in enumerate(source_dirs):
        with open(os.path.join(source_dir, "grapheme.json")) as file:
            phoneme_json = json.load(file)

        lookup = ["1_h", "2_e", "3_y", "4_s", "5_n", "6_i", "7_p", "8_s"]

        rows = []
        for filename, values in phoneme_json.items():
            for i, value in enumerate(values, start=1):
                rows.append((filename, lookup[i - 1], value))

        df = pd.DataFrame(rows, columns=["filename", "phoneme", "confidence"])

        normalized_df = {
            "filename": [],
            "phoneme": [],
            "confidence": [],
        }

        for ph in lookup:
            sub_df = df[df["phoneme"] == ph]

            mean_conf = sub_df["confidence"].mean()
            std = sub_df["confidence"].std()
            mean = sub_df["confidence"].mean()

            mean_to_one = sub_df["confidence"] / mean_conf
            std = mean_to_one.std()
            mean = mean_to_one.mean()
            max = mean_to_one.max()

            # limited = mean_to_one / (1 + mean_to_one.pow(2)).pow(0.5)
            limited = mean_to_one.clip(0, 1.0)
            std = limited.std()
            mean = limited.mean()
            max = limited.max()

            adjusted = mean_to_one / std
            std = adjusted.std()
            mean = adjusted.mean()

            normalized_df["filename"].extend(sub_df["filename"].tolist())
            normalized_df["phoneme"].extend([ph] * sub_df.shape[0])
            normalized_df["confidence"].extend(limited.tolist())

        df = pd.DataFrame(normalized_df)

        for ph in lookup:
            sub_df = df[df["phoneme"] == ph]
            print(f"mean for {ph}", sub_df["confidence"].mean())

        ax_idx = (idx // 2, idx % 2)

        # ax = axs[ax_idx]
        ax = axs

        # color_key = source_dir.split("/")[-1].split("_")[0]
        color_key = "ORG"

        sns.violinplot(data=df, x="phoneme", y="confidence", inner="quart", cut=0,
                       density_norm="width",
                       ax=ax,
                       color=color_palette[color_key], alpha=0.75, saturation=1.0)
        ax.set_ylim((0, 1))
        # ax.grid(True, which="major", linestyle="-")
        ax.set_xlabel("Grapheme")
        ax.set_ylabel("Confidence Score")

        # label = color_key + " - " + source_dir.split("/")[-1].split("_")[2]
        label = "ORG - Snips"

        if label == "VC - Sonos":
            label = "VC - Snips"

        ax.set_title(label)

    plt.tight_layout()
    # plt.title("Confidence per Grapheme")
    fig.savefig("Grapheme_Confidence.svg", format="svg")
    plt.show()


def plot_array():
    source_data = {
        "scores_good": [
            9.9649e-01, 9.6587e-01, 9.9992e-01, 9.9964e-01, 9.9554e-01, 9.9913e-01,
            9.9929e-01, 9.9929e-01, 9.9916e-01, 5.6619e-05, 9.9901e-01, 9.9896e-01,
            9.9904e-01, 9.9915e-01, 9.9911e-01, 9.9906e-01, 9.9895e-01, 9.9872e-01,
            9.9831e-01, 9.9427e-01, 9.6231e-01, 9.9745e-01, 9.9803e-01, 9.9816e-01,
            9.9851e-01, 9.9880e-01, 9.8726e-01, 9.9773e-01, 9.9035e-01, 9.9932e-01,
            9.9950e-01, 9.9922e-01, 9.7139e-01, 9.9186e-01, 9.9111e-01, 9.7938e-01,
            9.4855e-01, 9.8114e-01, 9.9889e-01, 9.9960e-01, 9.8944e-01, 9.9793e-01,
            9.9847e-01, 9.9780e-01, 9.9716e-01, 9.9792e-01, 9.9694e-01, 9.9637e-01,
            9.9625e-01, 9.9725e-01, 9.9838e-01, 9.9847e-01, 9.9297e-01
        ],
        "scores_missing": [
            1.0378e-04, 1.7587e-02, 1.3051e-03, 9.0646e-04, 1.2887e-03, 2.8623e-04,
            9.8739e-01, 9.9008e-01, 7.6053e-04, 9.9734e-01, 9.9945e-01, 9.9807e-01,
            7.9196e-04
        ],
        "scores_added": [
            0.9740, 0.7411, 0.9997, 0.9998, 0.9997, 0.0041, 0.9986, 0.9972, 0.9930,
            0.9879, 0.9832, 0.9911, 0.9864, 0.0147, 0.9872, 0.9888, 0.9632, 0.9623,
            0.9498, 0.9823, 0.0034, 0.9965, 0.9979, 0.9986, 0.9986, 0.0383, 0.9973,
            0.9949, 0.9884, 0.9679, 0.9486, 0.9861, 0.9988, 0.9990, 0.0022, 0.9929,
            0.9721, 0.9779, 0.9873, 0.9918, 0.9954, 0.0345, 0.6279, 0.9872, 0.9862,
            0.9890, 0.9959, 0.9972, 0.9977, 0.9974, 0.9977, 0.9984, 0.9982, 0.9985,
            0.9988, 0.9989, 0.9992, 0.9992, 0.9994, 0.9994, 0.9995, 0.9994, 0.9994,
            0.9994, 0.9994, 0.9994, 0.9994, 0.9993, 0.9993, 0.9995, 0.9996, 0.9996,
            0.9996, 0.9996, 0.9993, 0.9998, 0.9995, 0.9998, 0.9995, 0.9996, 0.9998,
            0.9995, 0.9996, 0.9997, 0.9994, 0.9994, 0.9993, 0.9994, 0.9996, 0.9997,
            0.9996, 0.9993, 0.9992, 0.9995, 0.9994, 0.9994, 0.9991, 0.9996, 0.9996,
            0.9996, 0.9996, 0.9996, 0.9997, 0.9997, 0.9996, 0.9997, 0.9997, 0.9998,
            0.9996, 0.9996, 0.9996, 0.9997, 0.9997, 0.9997, 0.9997, 0.9997, 0.9995,
            0.9997, 0.9997, 0.9998, 0.9998, 0.9998, 0.9998, 0.9997, 0.9998, 0.9998,
            0.9998, 0.9998, 0.9998, 0.9998, 0.9998, 0.9998, 0.9998, 0.9998, 0.9999,
            0.9998, 0.9999, 0.9998, 0.9997, 0.9998, 0.9998, 0.9998, 0.9998, 0.9998,
            0.9998, 0.9998, 0.9998, 0.9998, 0.9998, 0.9998, 0.9999, 0.9998, 0.9998,
            0.9998, 0.9998, 0.9998, 0.9999, 0.9998, 0.9999, 0.9999, 0.9998, 0.9999,
            0.9999, 0.9998, 0.9999, 0.9999, 0.9998, 0.9999, 0.9999, 0.9999, 0.9999,
            0.9999, 0.9999, 0.9999, 0.9998, 0.9998, 0.9998, 0.9998, 0.9999, 0.9999,
            0.9998, 0.9998, 0.9999, 0.9998, 0.9998, 0.9999, 0.9999, 0.9999, 0.9998,
            0.9998, 0.9998, 0.9998, 0.9998, 0.9998, 0.9998, 0.9999, 0.9998, 0.9999,
            0.9999, 0.9999, 0.9998, 0.9999, 0.9998, 0.9998, 0.9999, 0.9999, 0.9999,
            0.9999, 0.9999, 0.9998, 0.9999, 0.9998, 0.9998, 0.9998, 0.9998, 0.9998,
            0.9996, 0.9996, 0.9995, 0.9993, 0.0475, 0.1312, 0.9988, 0.9999, 0.9999,
            0.9999, 0.0107, 0.9996, 0.9996, 0.9996, 0.9995, 0.9992, 0.9992, 0.9991,
            0.9990, 0.9991, 0.9991, 0.9990, 0.9984, 0.9966, 0.9601, 0.9409, 0.9958,
            0.9954, 0.9944, 0.9947, 0.9958, 0.9894, 0.9947, 0.9964, 0.9967, 0.9962,
            0.9947, 0.9931, 0.9864, 0.9903, 0.9791, 0.9947, 0.9977, 0.9980, 0.9986,
            0.9988, 0.8719, 0.9978, 0.9966, 0.9890, 0.9852, 0.9666, 0.9532, 0.9961,
            0.9993, 0.9995, 0.9995, 0.9994, 0.9710, 0.9984, 0.9977, 0.9948, 0.9889,
            0.9883, 0.9907, 0.9930, 0.9953, 0.9966, 0.9507
        ],
        "tokens_good": [
            15, 0, 0, 0, 3, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 8, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 2, 0, 0, 0,
            0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8
        ],
        "tokens_missing": [
            15, 3, 16, 8, 4, 2, 0, 0, 18, 0, 0, 0, 8
        ],
        "tokens_added": [
            15, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8
        ]
    }

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    prefix_lookup = {
        "good": "Good",
        "added": "Added Speech",
        "missing": "Missing Wake Word",
    }

    for i, prefix in enumerate(["good", "added", "missing"]):
        confidence = source_data[f"scores_{prefix}"]
        tokens = source_data[f"tokens_{prefix}"]

        filler_confidence, phoneme_confidence = [], []
        for idx, y in enumerate(confidence):
            if tokens[idx] == 0:
                filler_confidence.append(y)
            else:
                phoneme_confidence.append(y)

        filler_confidence = np.array(filler_confidence)
        phoneme_confidence = np.array(phoneme_confidence)

        filler_confidence_mean = {
            "arith": filler_confidence.mean(),
            "gen": np.power((np.power(filler_confidence, -2).sum() / filler_confidence.shape[0]), 1 / -2)
        }

        phoneme_confidence_mean = {
            "arith": phoneme_confidence.mean(),
            "gen": np.power((np.power(phoneme_confidence, 2).sum() / phoneme_confidence.shape[0]), 1 / 2)
        }

        data = {
            "y": confidence,
            "x": np.arange(len(confidence)),
            "token_assigned": [1 if t != 0 else 0 for t in tokens]
        }

        df = pd.DataFrame(data)

        cmap = matplotlib.cm.get_cmap('Set2')
        print(cmap(0))
        print(cmap(1))

        print("arith graphe:", round(phoneme_confidence_mean["arith"], 3))
        print("arith filler:", round(filler_confidence_mean["arith"], 3))

        print("gen 2 graphe:", round(phoneme_confidence_mean["gen"], 3))
        print("gen -2 filler:", round(filler_confidence_mean["gen"], 3))

        print("arithmetic_total:", round(filler_confidence_mean["arith"] * phoneme_confidence_mean["arith"], 3))
        print("gen_total:", round(filler_confidence_mean["gen"] * phoneme_confidence_mean["gen"], 3))

        print("==================")

        sns.scatterplot(data=df, x="x", y="y", style="token_assigned", hue="token_assigned", palette="Set2",
                        ax=axs[i],
                        s=60)
        axs[i].set_title(prefix_lookup[prefix])
        axs[i].set_xlabel("Frames")
        axs[i].set_ylabel("Confidence")

        x_line = np.linspace(0, len(confidence), 2)
        y_line = np.array([phoneme_confidence_mean["arith"]] * 2)
        axs[i].plot(x_line, y_line, color=(0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
                    label="arit mean",
                    linestyle="dotted")

        x_line = np.linspace(0, len(confidence), 2)
        y_line = np.array([phoneme_confidence_mean["gen"]] * 2)
        axs[i].plot(x_line, y_line, color=(0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
                    label="gen mean (p=2)",
                    linestyle="dashed")

        x_line = np.linspace(0, len(confidence), 2)
        y_line = np.array([filler_confidence_mean["arith"]] * 2)
        axs[i].plot(x_line, y_line, color=(0.4, 0.7607843137254902, 0.6470588235294118),
                    label="arit mean", linestyle="dotted")

        x_line = np.linspace(0, len(confidence), 2)
        y_line = np.array([filler_confidence_mean["gen"]] * 2)
        axs[i].plot(x_line, y_line, color=(0.4, 0.7607843137254902, 0.6470588235294118),
                    label="gen mean (p=-2)", linestyle="dashed")

        # axs[i].legend(loc="right", labels=["empty", "grapheme carrying"])
        axs[i].legend().remove()

    # fig.legend(loc="lower center", ncols=2)
    fig.legend(loc="lower center", labels=["grapheme carrying", "empty"], ncols=2)

    plt.tight_layout(rect=[0, 0.08, 1, 1.0])
    plt.show()

    fig.savefig("FA_cs.svg", format="svg")
