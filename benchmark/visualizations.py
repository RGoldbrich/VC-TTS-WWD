# standard lib
import ast
import csv
import json
import math
import os
import re
from ast import literal_eval

# third party
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter
import numpy as np
import pylab as pl
import seaborn as sns
import sklearn.metrics as metrics
import statsmodels.api as sm
from sklearn.preprocessing import MultiLabelBinarizer

# application
from common.visualizations import plot_hist_from_df as plot_hist_from_df, COLORS as COLORS

name_simplification_dict = {
    "noisy_RNN": "RNN (Coucke et al.)",
    "noisy_WaveNet": "WaveNet (Coucke et al.)",
    "noisy_CNN": "CNN (Coucke et al.)",
    "clean_RNN": "RNN (Coucke et al.)",
    "clean_WaveNet": "WaveNet (Coucke et al.)",
    "clean_CNN": "CNN (Coucke et al.)",

    # for rising edge vs ind
    # "CRNN_98kp_100HUM_241_21_LFBE_W25_H10_Organic_Clean": "rising-edge fp",
    # "CRNN_98kp_100HUM_241_21_LFBE_W25_H10_Organic_CleanIND": "individual fp",
    # "CRNN_98kp_100HUM_241_21_LFBE_W25_H10_Organic_Noisy": "rising-edge fp",
    # "CRNN_98kp_100HUM_241_21_LFBE_W25_H10_Organic_NoisyIND": "individual fp",

    # for organic comp
    "CRNN_98kp_100HUM_241_21_LFBE_W25_H10_Organic_Clean": "CRNN (our)",
    "CRNN_98kp_100HUM_241_21_LFBE_W25_H10_Organic_Noisy": "CRNN (our)",

    # for org vs syn
    # "CRNN_98kp_100HUM_241_21_LFBE_W25_H10_Organic_Clean": "HUM",
    # "CRNN_98kp_100HUM_241_21_LFBE_W25_H10_Organic_Noisy": "HUM",
    "CRNN_241kp_100HUM_241_21_LFBE_W25_H10_OrganicL_Clean": "241k",
    "CRNN_241kp_100HUM_241_21_LFBE_W25_H10_OrganicL_Noisy": "241k",
    "CRNN_98kp_11HUM53VC37TTS_241_21_LFBE_W25_H10_OrganicPlus_Clean": "ORG+TTS+VC",
    "CRNN_98kp_11HUM53VC37TTS_241_21_LFBE_W25_H10_OrganicPlus_Noisy": "ORG+TTS+VC",
    "CRNN_98kp_59VC41TTS_241_21_LFBE_W25_H10_SynthBase_Clean": "TTS+VC",
    "CRNN_98kp_59VC41TTS_241_21_LFBE_W25_H10_SynthBase_Noisy": "TTS+VC",
    "CRNN_98kp_100VC_241_21_LFBE_W25_H10_VC_Clean": "VC",
    "CRNN_98kp_100VC_241_21_LFBE_W25_H10_VC_Noisy": "VC",
    "CRNN_98kp_100TTS_241_21_LFBE_W25_H10_TTS_Clean": "TTS",
    "CRNN_98kp_100TTS_241_21_LFBE_W25_H10_TTS_Noisy": "TTS",

    "CRNN_98kp_59VC41TTS_241_21_LFBE_W25_H10_SynthNF_Clean": "TTS+VC (unfiltered)",
    "CRNN_98kp_59VC41TTS_241_21_LFBE_W25_H10_SynthNF_Noisy": "TTS+VC (unfiltered)",
    "CRNN_98kp_59VC41TTS_241_21_LFBE_W25_H10_SynthCF_Clean": "TTS+VC (C-filtered)",
    "CRNN_98kp_59VC41TTS_241_21_LFBE_W25_H10_SynthCF_Noisy": "TTS+VC (C-filtered)",
    "CRNN_98kp_59VC41TTS_241_21_LFBE_W25_H10_SynthLF_Clean": "TTS+VC (D-filtered)",
    "CRNN_98kp_59VC41TTS_241_21_LFBE_W25_H10_SynthLF_Noisy": "TTS+VC (D-filtered)",

    "CRNN_98kp_59VC41TTS_241_21_LFBE_W25_H10_SynthNoRir_Clean": "TTS+VC (NoRir)",
    "CRNN_98kp_59VC41TTS_241_21_LFBE_W25_H10_SynthNoRir_Noisy": "TTS+VC (NoRir)",
    "CRNN_98kp_59VC41TTS_241_21_LFBE_W25_H10_SynthRir_Clean": "TTS+VC (Rir)",
    "CRNN_98kp_59VC41TTS_241_21_LFBE_W25_H10_SynthRir_Noisy": "TTS+VC (Rir)",

    "CRNN_98kp_59VC41TTS_241_21_LFBE_W25_H10_SynthNV_Clean": "TTS+VC (No VT)",
    "CRNN_98kp_59VC41TTS_241_21_LFBE_W25_H10_SynthNV_Noisy": "TTS+VC (No VT)",

    "CRNN_98kp_59VC41TTS_241_21_LFBE_W25_H10_AblWO_Clean": "TTS+VC (plain)",
    "CRNN_98kp_59VC41TTS_241_21_LFBE_W25_H10_AblWO_Noisy": "TTS+VC (plain)",

    "CRNN_98kp_70VC30TTS_241_21_LFBE_W25_H10_HailMary_Clean": "TTS+VC (Full)",
    "CRNN_98kp_70VC30TTS_241_21_LFBE_W25_H10_HailMary_Noisy": "TTS+VC (Full)",

    "CRNN_98kp_100VC_241_21_LFBE_W25_H10_Refer_Clean": "VC (Snips)",
    "CRNN_98kp_100VC_241_21_LFBE_W25_H10_Refer_Noisy": "VC (Snips)",
    "CRNN_98kp_100VC_241_21_LFBE_W25_H10_ReferS_Clean": "VC (Snips - 5k)",
    "CRNN_98kp_100VC_241_21_LFBE_W25_H10_ReferS_Noisy": "VC (Snips - 5k)",

    "CRNN_98kp_100VC_241_21_LFBE_W25_H10_MCV_Clean": "VC (Mcv)",
    "CRNN_98kp_100VC_241_21_LFBE_W25_H10_MCV_Noisy": "VC (Mcv)",
    "CRNN_98kp_100VC_241_21_LFBE_W25_H10_PeopleSpeech_Clean": "VC (PS)",
    "CRNN_98kp_100VC_241_21_LFBE_W25_H10_PeopleSpeech_Noisy": "VC (PS)",
    "CRNN_98kp_100VC_241_21_LFBE_W25_H10_TedLium_Clean": "VC (Ted)",
    "CRNN_98kp_100VC_241_21_LFBE_W25_H10_TedLium_Noisy": "VC (Ted)",

    "CRNN_98kp_100VC_241_21_LFBE_W25_H10_MCVAblWO_Clean": "VC (Mcv, plain)",
    "CRNN_98kp_100VC_241_21_LFBE_W25_H10_MCVAblWO_Noisy": "VC (Mcv, plain)",
    "CRNN_98kp_100VC_241_21_LFBE_W25_H10_PeopleSpeechAblWO_Clean": "VC (PS, plain)",
    "CRNN_98kp_100VC_241_21_LFBE_W25_H10_PeopleSpeechAblWO_Noisy": "VC (PS, plain)",
    "CRNN_98kp_100VC_241_21_LFBE_W25_H10_TedLiumAblWO_Clean": "VC (Ted, plain)",
    "CRNN_98kp_100VC_241_21_LFBE_W25_H10_TedLiumAblWO_Noisy": "VC (Ted, plain)",

    "CRNN_98kp_100TTS_241_21_LFBE_W25_H10_TTSNoRIR_Noisy": "TTS (NoRir)",
    "CRNN_98kp_100TTS_241_21_LFBE_W25_H10_TTSNV_Noisy": "TTS (No VT)",
    "CRNN_98kp_100TTS_241_21_LFBE_W25_H10_TTSNF_Noisy": "TTS (unfiltered)",
    "CRNN_98kp_100TTS_241_21_LFBE_W25_H10_TTSAblWO_Noisy": "TTS (plain)",
    "CRNN_98kp_100TTS_241_21_LFBE_W25_H10_TTSAblWO_Clean": "TTS (plain)",

    "CRNN_98kp_59VC41TTS_241_21_LFBE_W25_H10_SynthPart_Noisy": "TTS+VC (partials)",
    "CRNN_98kp_59VC41TTS_241_21_LFBE_W25_H10_SynthPart_Clean": "TTS+VC (partials)",
}

palette_dict = {
    "CNN (Coucke et al.)": "#4d87e3",
    "RNN (Coucke et al.)": "#d1b149",
    "WaveNet (Coucke et al.)": "#7b45b5",
    "CRNN (our)": "#8c9c54",

    "rising-edge fp": "#e84a0c",
    "individual fp": "#0c76e8",

    "HUM": "#8c9c54",  # earthy, olive, moss
    "ORG+TTS+VC": "#c9951c",
    "TTS+VC": "#b83064",  # red deep purple
    "TTS": "#305582",  # blue
    "VC": "#F9A602",  # yellow sunrise

    "TTS+VC (proj)": "#e36696",  # red deep purple
    "TTS (proj)": "#5680b3",  # blue
    "VC (proj)": "#f0c87a",  # yellow sunrise

    "TTS+VC (unfiltered)": "#f590b7",
    "TTS+VC (C-filtered)": "#ff0000",
    "TTS+VC (D-filtered)": "#0000ff",

    "TTS+VC (NoRir)": "#0e899e",
    "TTS+VC (Rir)": "#169e56",

    "TTS+VC (No VT)": "#f590b7",

    "TTS+VC (plain)": "#f590b7",

    "TTS (plain)": "#8299b5",

    "TTS+VC (Full)": "#247e96",
    "TTS+VC (Full) (proj)": "#4798ad",

    "VC (Snips)": "#d1681d",
    "VC (Snips - 5k)": "#124eb5",

    "VC (Mcv)": "#b5483f",
    "VC (PS)": "#3fab70",
    "VC (Ted)": "#4578ba",

    "VC (Mcv, plain)": "#eb9e98",
    "VC (PS, plain)": "#87c7a4",
    "VC (Ted, plain)": "#83aee6",

    "99k": "#8c9c54",
    "241k": "#9c5454",

    "TTS+VC (partials)": "#f590b7",
}


def yield_train_out_folders(source_dir: str):
    """
    Generator function that yields available information about train or benchmark output folders
    :param source_dir: parent directory of train or benchmark output folders
    """
    folders = sorted(os.listdir(source_dir))

    model_count = dict()

    for folder in folders:
        if folder.startswith("_"):
            continue

        if not re.search("\d{6}_\d{6}", folder):
            continue

        # path of test and train info jsons if present
        test_info_path = os.path.join(source_dir, folder, "test_info.json") \
            if "test_info.json" in os.listdir(os.path.join(source_dir, folder)) else None

        train_info_path = os.path.join(source_dir, folder, "train_info.json") \
            if "train_info.json" in os.listdir(os.path.join(source_dir, folder)) else None

        db_tags_csv = os.path.join(source_dir, folder, "db_and_tags.csv") \
            if "db_and_tags.csv" in os.listdir(os.path.join(source_dir, folder)) else None

        # architecture stub of model
        sep_loc = [m.start() for m in re.finditer("_", folder)]

        model_name = folder[sep_loc[1] + 1:sep_loc[-4] if folder[-2:] == "OF" else sep_loc[
            -3]]  # CRNN_103kp_100HUM_121_25_LFBE_W40_H20_Clean
        model_type = folder[sep_loc[1] + 1:sep_loc[2]]  # CRNN

        if model_name not in model_count:
            model_count[model_name] = 0

        if test_info_path is not None:
            yield (model_name,  # CRNN_103kp_100HUM_121_25_LFBE_W40_H20_Clean
                   model_type,  # CRNN
                   model_count[model_name],  # 0
                   os.path.join(source_dir, folder),  # <full folder path>
                   train_info_path,  # <path to train info json>
                   test_info_path,  # <path to test info json>
                   db_tags_csv)  # <path to db and tags json>

            model_count[model_name] += 1


def train_df_from_folder(source_dir: str) -> pd.DataFrame:
    """
    Load all train info json from all train subdirectories into single pandas dataframe.
    :param source_dir: training parent directory
    :return: dataframe containing all information present in the train info json
    """
    data = {
        "model_name": [],
        "model_type": [],
        "train_i": [],
        "epoch": [],
        "train_tp": [],
        "valid_tp": [],
        "train_f1": [],
    }

    # get folders
    for (model_name, model_type, it, _, train_info_path, _, _) in yield_train_out_folders(source_dir):
        if train_info_path is None:
            continue

        # load train info json
        with open(train_info_path, "r") as f:
            train_info = json.load(f)

        data["model_name"].append(model_name)
        data["model_type"].append(model_type)
        data["train_i"].append(it)
        data["epoch"].append(train_info["best_model"]["epoch"])
        data["train_tp"].append(train_info["best_model"]["train_metrics"]["tp_rate_recall_sens"])
        data["valid_tp"].append(train_info["best_model"]["validation_metrics"]["tp_rate_recall_sens"])
        data["train_f1"].append(train_info["best_model"]["validation_metrics"]["f1_score"])

    return pd.DataFrame(data)


def find_index_with_tolerance(L, f, tolerance=1e-7):
    for i, val in enumerate(L):
        if abs(f - val) < tolerance:
            return i

    raise ValueError("Index not found")


def test_df_from_folder(source_dir: str, aux_only=False) -> pd.DataFrame:
    """
    Load all test info json from all train subdirectories into single pandas dataframe.
    :param source_dir: test parent directory
    :return: dataframe containing all information present in the test info json
    """
    if aux_only:
        data = {
            "model_name": [],
            "model_type": [],
            "train_i": [],
            "fixed_fp_threshold": [],
            "auc_roc": [],
            "auc_roc_three_min": [],
            "auc_roc_three_mean": [],
            "test_tp": [],
        }
    else:
        data = {
            "model_name": [],
            "model_type": [],
            "train_i": [],
            "threshold": [],
            "tp": [],
            "fp": [],
        }

    min_or_mean = "accepts_rel_three_mean"
    # min_or_mean = "accepts_rel_three_min"

    for (model_name, model_type, train_i, _, _, test_info_path, _) in yield_train_out_folders(source_dir):
        # load test info json
        with open(test_info_path, "r") as f:
            test_info = json.load(f)

        n_thresholds = len(test_info["combined_true_accepts"][min_or_mean])
        if aux_only:
            data["model_name"].append(model_name)
            data["model_type"].append(model_type)
            data["train_i"].append(train_i)
            data["fixed_fp_threshold"].append(float(test_info["aux"]["fixed_fp_threshold"]))
            data["auc_roc"].append(float(test_info["aux"]["auc_roc"]))
            data["auc_roc_three_min"].append(float(test_info["aux"]["auc_roc_three_min"]))
            data["auc_roc_three_mean"].append(float(test_info["aux"]["auc_roc_three_mean"]))

            thresholds = test_info["thresholds"]
            f = float(test_info["aux"]["fixed_fp_threshold"])
            index = find_index_with_tolerance(thresholds, float(test_info["aux"]["fixed_fp_threshold"]))
            val = thresholds[index]

            data["test_tp"].append(test_info["combined_true_accepts"][min_or_mean][index])
        else:
            data["model_name"].extend([model_name] * n_thresholds)
            data["model_type"].extend([model_type] * n_thresholds)
            data["train_i"].extend([train_i] * n_thresholds)
            data["threshold"].extend(test_info["thresholds"])
            data["tp"].extend(test_info["combined_true_accepts"][min_or_mean])
            data["fp"].extend(test_info["combined_false_accepts"][min_or_mean])

    return pd.DataFrame(data)


def db_tags_df_from_folder(source_dir: str) -> pd.DataFrame:
    data = {
        "model_name": [],
        "model_type": [],
        "train_i": [],
        "decision_boundary": [],
        "tags": [],
    }

    for (mname, mtype, ti, _, _, _, db_tags_csv) in yield_train_out_folders(source_dir):
        # load db and tags info csv
        db_and_tags = pd.read_csv(db_tags_csv, index_col=0)

        n_files = db_and_tags.shape[0]
        data["model_name"].extend([mname] * n_files)
        data["model_type"].extend([mtype] * n_files)
        data["train_i"].extend([ti] * n_files)
        data["decision_boundary"].extend(db_and_tags["decision_boundary"])
        data["tags"].extend(db_and_tags["tags"])

    return pd.DataFrame(data)


def folder_to_roc(source_dir: str, filt: list[str] = None, comparison_csv_paths: list[str] = None, ax=None,
                  t=None) -> None:
    """
    Loads benchmark information from all output directories in given benchmark parent directory.
    Groups models by name, interpolates and averages tp rates.
    :param source_dir: benchmark parent directory to load
    :param filt: list of model names to filter for
    :param comparison_csv_paths: list of paths to tp/fp pairs csv files to plot as well
    """
    df = test_df_from_folder(source_dir)

    if filt:
        condition = df["model_name"].isin(filt)
        df = df[condition]

    # interpolate fp to be able to compute average of curves
    interp_data = {
        "model_name": [],
        "model_type": [],
        "train_i": [],
        "tp": [],
        "fp": [],
    }

    max_fp_encountered = 2.0
    fp_inter_resolution = 0.001

    fixed_fp = [0.133, 0.5, 1.0]
    fixed_fp_idx = [int((max_fp_encountered - f_id) / fp_inter_resolution) for f_id in fixed_fp]

    baseline = None

    for model in df["model_name"].unique():
        model_data = df[df["model_name"] == model]

        tp_rates = []
        zero_fp_tp_rates = []

        for train_i in model_data["train_i"].unique():
            train_data = model_data[model_data["train_i"] == train_i]

            # range of fixed fp-rates to interpolate
            fp_range = np.arange(0.00, max_fp_encountered + fp_inter_resolution, fp_inter_resolution)

            # find max in train_data["fp"] and discard values before that
            index_max = train_data["fp"].idxmax()
            fp_series = train_data["fp"][train_data["fp"].index >= index_max]
            tp_series = train_data["tp"][train_data["tp"].index >= index_max]

            # compute interpolated tp-rates
            tp_inter = np.interp(fp_range, max_fp_encountered - fp_series, tp_series)

            # build new dataframe with interpolated data
            n_entries = len(tp_inter)
            interp_data["model_name"].extend([model] * n_entries)
            interp_data["model_type"].extend([train_data["model_type"].unique()[0]] * n_entries)
            interp_data["train_i"].extend([train_i] * n_entries)
            interp_data["tp"].extend(tp_inter.tolist())
            interp_data["fp"].extend((max_fp_encountered - fp_range).tolist())

            # tp_rates.append(tp_inter.tolist()[fixed_fp_idx])
            tp_rates.append([tp_inter.tolist()[f_id] for f_id in fixed_fp_idx])

            # compute zero tolerance accept rate
            sub = train_data[train_data["fp"] == 0.0].sort_values(by="threshold", ascending=True)
            if sub.shape[0] > 0:
                zero_fp_tp_rates.append(sub.iloc[0]["tp"])

        if filt and model in filt:
            if not baseline:
                baseline = tp_rates

            print(f"{model}:  ({len(tp_rates)})")

            for idx, fp_rate in enumerate(fixed_fp):
                tp_s = pd.Series([rates[idx] for rates in tp_rates])
                print(f"  far: {fp_rate:.3f}")
                print(f"    raw:    {1 - tp_s.mean():.6f} || {tp_s.std():.6f}")
                print(f"    pretty: {1 - tp_s.mean():.2%} || {tp_s.std():.2%}")
                print(f"    range: {tp_s.max() - tp_s.min():.2%}")
                print(f"    zerofp_tp: {sum(zero_fp_tp_rates) / len(zero_fp_tp_rates):.2%}")

            for idx, fp_rate in enumerate(fixed_fp):
                tp_s = pd.Series([rates[idx] for rates in tp_rates])
                print(f"{(1 - tp_s.mean()) * 100:.2f}", end="")
                if idx < 2:
                    print(" & ", end="")
                else:
                    print("\t\t", end="")

            if baseline:
                for idx, fp_rate in enumerate(fixed_fp):
                    baseline_s = pd.Series([bs[idx] for bs in baseline])
                    tp_s = pd.Series([rates[idx] for rates in tp_rates])
                    means = baseline_s.mean(), tp_s.mean()
                    rel_change = ((1 - tp_s.mean()) - (1 - baseline_s.mean())) / (1 - baseline_s.mean())
                    print(f"{rel_change * 100:+.1f}\\%", end="")
                    if idx < 2:
                        print(" & ", end="")
                    else:
                        print("")

    # load comparison if provided
    if comparison_csv_paths:
        for csv_path in comparison_csv_paths:
            df = pd.read_csv(csv_path, index_col=None)

            n_entries = len(df)

            comparison_name = csv_path.split('/')[-1][:-12]
            if filt:
                filt.append(comparison_name)

            interp_data["model_name"].extend([comparison_name] * n_entries)
            interp_data["model_type"].extend([comparison_name.split('_')[-1]] * n_entries)
            interp_data["train_i"].extend([0] * n_entries)
            interp_data["tp"].extend(1.0 - df[" y"])  # be aware of the space here
            interp_data["fp"].extend(df["x"])

    interp_df = pd.DataFrame(interp_data)

    # apply filter and simplification
    print("Unique model names:\n", interp_df["model_name"].unique())
    if filt:
        condition = interp_df["model_name"].isin(filt)
        interp_df = interp_df[condition]

        interp_df["simplified_name"] = interp_df["model_name"].map(name_simplification_dict)
        interp_df["is_comparison"] = interp_df["model_type"] != "CRNN"

    # reverse tp to store fn instead
    interp_df["frr"] = 1.0 - interp_df["tp"]

    # plot
    if not ax:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.subplots()

    ax.set_xticks(np.arange(0, max_fp_encountered, 0.25))
    ax.set_xticks(np.arange(0, max_fp_encountered, 0.05), minor=True)

    if t == "Clean":
        ax.set_yticks(np.arange(0.0, 0.4, 0.02))
        ax.set_yticks(np.arange(0.0, 0.4, 0.004), minor=True)
    else:
        ax.set_yticks(np.arange(0.0, 0.4, 0.05))
        ax.set_yticks(np.arange(0.0, 0.4, 0.01), minor=True)

    # ax.set_yticks(np.arange(0.0, 0.5, 0.05))
    # ax.set_yticks(np.arange(0.0, 0.5, 0.01), minor=True)

    ax.grid(axis='x', which="major", linestyle=(0, (4, 6)))
    ax.grid(axis='y', which="major", linestyle=(0, (4, 6)))
    ax.grid(axis='y', which="minor", linestyle=(0, (1, 6)))

    offset = 0
    # sns.lineplot(data=interp_df, x="fp", y="frr", hue="simplified_name", estimator='mean',
    #              errorbar=None, palette=palette_dict, ax=ax, linewidth=2)
    # hue_order=[
    #     "VC (Mcv)", "VC (PS)", "VC (Ted)",
    #     "VC (Mcv, plain)", "VC (PS, plain)", "VC (Ted, plain)",
    # ])

    offset = 0
    sns.lineplot(data=interp_df, x="fp", y="frr", hue="simplified_name", estimator="mean",
                 errorbar=("pi", 100), palette=palette_dict, ax=ax, linewidth=2, err_style="band")

    # offset = 1
    # sns.lineplot(data=interp_df, x="fp", y="frr", hue="simplified_name", style="train_i",
    #              errorbar=None, palette=palette_dict, ax=ax, linewidth=2)

    ax.set_ylim((0.0, 0.25 if t == "Noisy" else 0.1))
    # ax.set_ylim((0.0, 0.4))
    ax.set_xlim((0, max_fp_encountered))
    ax.set_xlabel("False Accept Rate [per hour]")
    ax.set_ylabel("False Reject Rate")

    ax.yaxis.set_major_formatter(PercentFormatter(1))

    if t:
        ax.set_title(t)

    h, l = ax.get_legend_handles_labels()
    print(l)
    lim = len(filt)
    ax.legend(h[offset:lim + 1], l[offset:lim + 1], bbox_to_anchor=(0.5, -0.15), loc="upper center",
              fontsize="small", ncol=2)

    # plt.show()


def db_and_tags(source_dir: str, filt: list[str] = None, ax=None, title=None) -> None:
    df = db_tags_df_from_folder(source_dir)
    df["tags"] = df["tags"].apply(ast.literal_eval)

    aux_df = test_df_from_folder(source_dir, True)

    data_for_cat = {
        "subset": [],
        "model_name": [],
        "tp_rate": [],
        "pred": [],
    }

    for model in df["model_name"].unique():
        model_data = df[df["model_name"] == model]

        if filt and model not in filt:
            continue

        # total, accepts, sum of dbs
        mean_perf = {
            "Overall": [0, 0, 0],
            "TagFree": [0, 0, 0],
            "Emotional": [0, 0, 0],
            "VoiceAccent": [0, 0, 0],
            "Pronunciation": [0, 0, 0],
            "BadEnvironment": [0, 0, 0],
            # "unintelligible_quiet": [0, 0, 0],
        }

        for train_i in model_data["train_i"].unique():
            train_data = model_data[model_data["train_i"] == train_i]

            threshold = aux_df[(aux_df["model_name"] == model) & (aux_df["train_i"] == train_i)][
                "fixed_fp_threshold"].item()

            train_data = train_data[train_data["tags"].apply(lambda tags: "UnintelligibleQuiet" not in tags)]
            # train_data = train_data[train_data["tags"].apply(lambda tags: "BadEnvironment" not in tags)]
            mean_perf["Overall"][0] += train_data.shape[0]
            mean_perf["Overall"][1] += (train_data["decision_boundary"] > threshold).sum()
            mean_perf["Overall"][2] += train_data["decision_boundary"].sum()

            df_tag_free = train_data[train_data["tags"].apply(lambda tags: tags == [])]
            mean_perf["TagFree"][0] += df_tag_free.shape[0]
            mean_perf["TagFree"][1] += (df_tag_free["decision_boundary"] > threshold).sum()
            mean_perf["TagFree"][2] += df_tag_free["decision_boundary"].sum()

            df_emotional = train_data[train_data["tags"].apply(lambda tags: "Emotional" in tags)]
            mean_perf["Emotional"][0] += df_emotional.shape[0]
            mean_perf["Emotional"][1] += (df_emotional["decision_boundary"] > threshold).sum()
            mean_perf["Emotional"][2] += df_emotional["decision_boundary"].sum()

            df_voice = train_data[train_data["tags"].apply(lambda tags: "VoiceAccent" in tags)]
            mean_perf["VoiceAccent"][0] += df_voice.shape[0]
            mean_perf["VoiceAccent"][1] += (df_voice["decision_boundary"] > threshold).sum()
            mean_perf["VoiceAccent"][2] += df_voice["decision_boundary"].sum()

            df_pronunciation = train_data[train_data["tags"].apply(lambda tags: "Pronunciation" in tags)]
            mean_perf["Pronunciation"][0] += df_pronunciation.shape[0]
            mean_perf["Pronunciation"][1] += (df_pronunciation["decision_boundary"] > threshold).sum()
            mean_perf["Pronunciation"][2] += df_pronunciation["decision_boundary"].sum()

            df_env = train_data[train_data["tags"].apply(lambda tags: "BadEnvironment" in tags)]
            mean_perf["BadEnvironment"][0] += df_env.shape[0]
            mean_perf["BadEnvironment"][1] += (df_env["decision_boundary"] > threshold).sum()
            mean_perf["BadEnvironment"][2] += df_env["decision_boundary"].sum()

            # df_quiet = train_data[train_data["tags"].apply(lambda tags: "UnintelligibleQuiet" in tags)]
            # mean_perf["unintelligible_quiet"][0] += df_quiet.shape[0]
            # mean_perf["unintelligible_quiet"][1] += (df_quiet["decision_boundary"] > threshold).sum()
            # mean_perf["unintelligible_quiet"][2] += df_quiet["decision_boundary"].sum()

        print("============================================================================")
        print(f"{model} ({model_data['train_i'].max() + 1})")
        print(f"{'total':>32} {'tp':>8} {'tp%':>8} {'frr%':>8} {'pred':>8} {'rel. m.':>8}")

        frr_overall = 1 - mean_perf["Overall"][1] / mean_perf["Overall"][0]
        for key in mean_perf.keys():
            frr_key = 1 - mean_perf[key][1] / mean_perf[key][0]
            print(f"  {key:<20}: "
                  f"{mean_perf[key][0]:>8} "
                  f"{mean_perf[key][1]:>8} "
                  f"{mean_perf[key][1] / mean_perf[key][0]:>8.2%} "
                  f"{1 - mean_perf[key][1] / mean_perf[key][0]:>8.2%} "
                  f"{mean_perf[key][2] / mean_perf[key][0]:>8.2%}"
                  f"{(frr_key - frr_overall) / frr_overall:>+8.1%}")

            if model not in ["CRNN_98kp_100TTS_241_21_LFBE_W25_H10_Demand"]:
                data_for_cat["subset"].append(key)
                data_for_cat["model_name"].append(model)
                data_for_cat["tp_rate"].append(mean_perf[key][1] / mean_perf[key][0])
                data_for_cat["pred"].append(mean_perf[key][2] / mean_perf[key][0])

    df = pd.DataFrame(data_for_cat)
    df["simplified_name"] = df["model_name"].map(name_simplification_dict)
    df["frr"] = 1 - df["tp_rate"]

    # baseline = "TTS+VC"
    baseline = df["simplified_name"][0]

    # names = ["TTS+VC"]
    names = df[df["simplified_name"] != baseline]["simplified_name"].unique()
    # names = [n for n in names if n not in ["TTS+VC", "TTS+VC (Full)"]]
    # names = []
    # names = ["TTS"]
    # names = ["VC"]
    # names = ["TTS+VC", "VC"]
    # names = ["TTS+VC", "TTS", "VC"]

    # compute proj. base on ration in overall frr
    overall_frr = df[df["subset"] == "Overall"].set_index("simplified_name")["frr"]
    ratios = {name: overall_frr[name] / overall_frr[baseline] for name in names}

    projected_rows = []
    for subset in ["TagFree", "Emotional", "VoiceAccent", "Pronunciation", "BadEnvironment"]:
        org_frr = df[(df["simplified_name"] == baseline) & (df["subset"] == subset)]["frr"].values[0]
        for name in names:
            projected_rows.append(
                {"simplified_name": f"{name} (proj)", "subset": subset, "frr": org_frr * ratios[name]})

    df = pd.concat([df, pd.DataFrame(projected_rows)], ignore_index=True)

    # org_tp = df[df["simplified_name"] == "ORG"][["subset", "frr"]]
    # org_tp = org_tp.rename(columns={"frr": "frr_comp"})
    # df = df.merge(org_tp, on="subset", how="left")
    #
    # df["frr_rel_org"] = (df["frr"] - df["frr_comp"]) / df["frr_comp"]

    if not ax:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot()

    hue_order = ["HUM", "TTS+VC", "TTS+VC (proj)", "TTS", "TTS (proj)", "VC", "VC (proj)", "TTS+VC (Full)",
                 "TTS+VC (Full) (proj)"]
    hue_order = [t for t in hue_order if t in df["simplified_name"].unique()]

    sns.barplot(df, x="subset", y="frr", hue="simplified_name", palette=palette_dict, ax=ax,
                hue_order=hue_order,
                order=["Overall", "TagFree", "Emotional", "VoiceAccent", "Pronunciation", "BadEnvironment"])

    h, l = ax.get_legend_handles_labels()
    lim = len(filt)
    adding = 0
    adding = 2
    ax.legend(h[0:lim + 1 + adding], l[0:lim + 1 + adding], bbox_to_anchor=(0.5, -0.3), loc="upper center",
              fontsize="small", ncol=3)

    max_frr = df["frr"].max()

    ax.set_yticks(np.arange(0.0, 1.0, 0.01))
    ax.set_yticks(np.arange(0.0, 1.0, 0.0025), minor=True)
    ax.set_ylim((0.0, 0.05))

    if max_frr >= 0.05:  # towards 10
        ax.set_yticks(np.arange(0.0, 1.0, 0.025))
        ax.set_yticks(np.arange(0.0, 1.0, 0.005), minor=True)  # noisy syn
        ax.set_ylim((0.0, 0.10))
    if max_frr >= 0.10:  # towards 15
        ax.set_yticks(np.arange(0.0, 1.0, 0.05))
        ax.set_yticks(np.arange(0.0, 1.0, 0.01), minor=True)  # noisy syn
        ax.set_ylim((0.0, 0.15))
    if max_frr >= 0.15:  # towards 20
        ax.set_yticks(np.arange(0.0, 1.0, 0.05))
        ax.set_yticks(np.arange(0.0, 1.0, 0.01), minor=True)  # clean tts
        ax.set_ylim((0.0, 0.20))
    if max_frr >= 0.20:  # towards 30
        ax.set_yticks(np.arange(0.0, 1.0, 0.05))
        ax.set_yticks(np.arange(0.0, 1.0, 0.025), minor=True)  # clean tts
        ax.set_ylim((0.0, 0.30))
    if max_frr >= 0.30:  # towards 50
        ax.set_yticks(np.arange(0.0, 1.0, 0.1))
        ax.set_yticks(np.arange(0.0, 1.0, 0.025), minor=True)  # clean tts
        ax.set_ylim((0.0, 0.50))
    if max_frr >= 0.50:  # towards 75
        ax.set_yticks(np.arange(0.0, 1.0, 0.15))
        ax.set_yticks(np.arange(0.0, 1.0, 0.05), minor=True)  # clean tts
        ax.set_ylim((0.0, 0.75))

    ax.yaxis.set_major_formatter(PercentFormatter(1))

    ax.grid(axis='y', which="major", linestyle=(0, (4, 6)))
    ax.grid(axis='y', which="minor", linestyle=(0, (1, 10)))

    ax.set_xlabel("Subset")
    ax.set_ylabel("FRR at FAR of 0.5 per hour")
    if title:
        ax.set_title(title)

    ax.tick_params(axis='x', labelrotation=15)


def folder_to_swarm(source_dir: str, mode: str, filt: list[str] = None) -> None:
    # get train and test info dataframes
    train_df = train_df_from_folder(source_dir)
    test_df = test_df_from_folder(source_dir, aux_only=True)

    test_df_full = test_df_from_folder(source_dir)

    if train_df.empty:
        combined_df = test_df
    else:
        combined_df = pd.merge(train_df, test_df)

    if filt:
        condition = combined_df["model_name"].isin(filt)
        combined_df = combined_df[condition]

    interp_data = {
        "model_name": [],
        "model_type": [],
        "train_i": [],
        "tp": [],
        "fp": [],
    }

    max_fp_encountered = 1.5
    fp_inter_resolution = 0.002
    fixed_fp = 0.5
    fixed_fp_idx = int((max_fp_encountered - fixed_fp) / fp_inter_resolution)

    tp_rates = {
        "model_name": [],
        "train_i": [],
        "tp_rate": [],
    }
    for model in test_df_full["model_name"].unique():
        model_data = test_df_full[test_df_full["model_name"] == model]

        for train_i in model_data["train_i"].unique():
            train_data = model_data[model_data["train_i"] == train_i]

            # range of fixed fp-rates to interpolate
            fp_range = np.arange(0.00, max_fp_encountered + fp_inter_resolution, fp_inter_resolution)

            # find max in train_data["fp"] and discard values before that
            index_max = train_data["fp"].idxmax()
            fp_series = train_data["fp"][train_data["fp"].index >= index_max]
            tp_series = train_data["tp"][train_data["tp"].index >= index_max]

            # compute interpolated tp-rates
            tp_inter = np.interp(fp_range, max_fp_encountered - fp_series, tp_series)
            tp_rates["model_name"].append(model)
            tp_rates["train_i"].append(train_i)
            tp_rates["tp_rate"].append(tp_inter.tolist()[fixed_fp_idx])

    baseline = None

    for model in combined_df["model_name"].unique():
        model_data = combined_df[combined_df["model_name"] == model]
        print(f"{model}")
        print(
            f"aucroc: {model_data['auc_roc_three_min'].mean():.10f} ({model_data['auc_roc_three_min'].std():.10f})")
        print(f"aucdet: {(1 - model_data['auc_roc_three_min'].mean()):.10f}")

        print(
            f"aucroc: {model_data['auc_roc_three_min'].mean() * 1e6:.1f} ({model_data['auc_roc_three_min'].std() * 1e6:.1f})")
        print(
            f"aucdet: {(1 - model_data['auc_roc_three_min'].mean()) * 1e6:.1f} ({model_data['auc_roc_three_min'].std() * 1e6:.1f})")

        if not baseline:
            baseline = 1 - model_data['auc_roc_three_min'].mean()

        print(f"{((1 - model_data['auc_roc_three_min'].mean()) - baseline) / baseline * 100:+.1f}\\%")

        if "train_tp" in model_data:
            print(f"frr on train: {1 - model_data['train_tp'].mean():.2%} ({model_data['train_tp'].std():.2})")
            print(f"f1 on train: {model_data['train_f1'].mean():.2%} ({model_data['train_f1'].std():.2})\n")

    tp_rate_df = pd.DataFrame(tp_rates, columns=["model_name", "train_i", "tp_rate"])

    combined_df = pd.merge(combined_df, tp_rate_df, on=["model_name", "train_i"])

    combined_df["simplified_name"] = combined_df["model_name"].map(name_simplification_dict)


def mean_and_std(source_dir: str, val: str, filt: list[str] = None) -> None:
    train_df = train_df_from_folder(source_dir)

    test_df = test_df_from_folder(source_dir, aux_only=True)

    if train_df.empty:
        combined_df = test_df
    else:
        combined_df = pd.merge(train_df, test_df)

    if filt:
        condition = combined_df["model_name"].isin(filt)
        combined_df = combined_df[condition]

    for model in combined_df["model_name"].unique():
        model_data = combined_df[combined_df["model_name"] == model]

        print(f"{model}:")

        print(f"  mean: {model_data[val].mean():.4f}")
        print(f"  std:  {model_data[val].std():.6f}")

        # auc
        print(f"  pretty:  {model_data[val].mean():5f}(+-{model_data[val].std():.5f})")

        # to
        print(f"  pretty:  {model_data[val].mean():.1%}")


def plot_sklearn_det(source_dir: str) -> None:
    expected = np.load(f"{source_dir}/expected.npy")
    three_min = np.load(f"{source_dir}/predicted_min.npy")

    expected_pos = three_min[expected > 0.5]

    fig = plt.figure(figsize=(36, 24))
    ax = fig.add_subplot()

    metrics.DetCurveDisplay.from_predictions(expected, three_min, ax=ax)

    plt.tight_layout()
    plt.show()
