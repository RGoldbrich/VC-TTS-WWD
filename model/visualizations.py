# standard lib
import json
import os
import re

# third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec

# application
from common.metrics import compute_metrics_from_cm, get_keys


def vis_training_process(source_dir: str) -> None:
    df = pd.read_csv(source_dir + "/info.csv", index_col=None)
    print("columns:", df.columns.tolist())

    train_metric_list = [compute_metrics_from_cm(np.int64(row["train_tn"]),
                                                 np.int64(row["train_fp"]),
                                                 np.int64(row["train_fn"]),
                                                 np.int64(row["train_tp"]), ) for _, row in df.iterrows()]

    valid_metric_list = [compute_metrics_from_cm(np.int64(row["valid_tn"]),
                                                 np.int64(row["valid_fp"]),
                                                 np.int64(row["valid_fn"]),
                                                 np.int64(row["valid_tp"]), ) for _, row in df.iterrows()]

    test_metric_list = [compute_metrics_from_cm(np.int64(0),
                                                np.int64(0),
                                                np.int64(row["test_fn"]),
                                                np.int64(row["test_tp"]), ) for _, row in df.iterrows()]

    epochs = list(range(df.shape[0]))

    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[3, 1])
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_lr = fig.add_subplot(gs[1, 0])
    ax_metrics = fig.add_subplot(gs[:, 1])

    fig.suptitle(source_dir.split('/')[-1])

    # loss
    ax_loss.plot(epochs, df["epoch_mean_loss"], label="epoch_mean_loss")
    # ax_loss.plot(epochs, df["epoch_mean_train_loss"], label="epoch_mean_train_loss")
    ax_loss.plot(epochs, df["epoch_mean_validation_loss"], label="epoch_mean_validation_loss")
    ax_loss.plot(epochs, df["best_validation_loss"], label="best_validation_loss")

    ax_loss.set_title("loss")
    ax_loss.set_yscale("log")
    ax_loss.grid()
    ax_loss.legend()

    # learn rate
    ax_lr.set_title("learn rate")
    ax_lr.plot(epochs, df["lr"], label="lr", )
    ax_lr.set_yscale("log")
    ax_lr.grid()

    # metrics
    ax_metrics.set_title("metrics")
    ax_metrics.plot(epochs, [m["tp_rate_recall_sens"] for m in train_metric_list], label="train-tp")
    # ax_metrics.plot(epochs, [m["tn_rate_spec"] for m in train_metric_list], label="tra-reject")
    ax_metrics.plot(epochs, [m["f1_score"] for m in train_metric_list], label="train-f1")

    ax_metrics.plot(epochs, [m["tp_rate_recall_sens"] for m in valid_metric_list], label="val-tp")
    # ax_metrics.plot(epochs, [m["tn_rate_spec"] for m in valid_metric_list], label="val-reject")
    ax_metrics.plot(epochs, [m["f1_score"] for m in valid_metric_list], label="val-f1")

    ax_metrics.plot(epochs, [m["tp_rate_recall_sens"] for m in test_metric_list], label="accept-human")

    ax_metrics.grid()
    ax_metrics.legend()

    plt.tight_layout()
    plt.show()


def vis_folder(source_dir: str) -> None:
    folders = sorted(os.listdir(source_dir))

    gen_data = {
        "model_desc": [],
    }

    train_data_mid_threshold = {key: [] for key in get_keys()}
    valid_data_mid_threshold = {key: [] for key in get_keys()}
    test_data_fixed_fp_all = {key: [] for key in get_keys()}
    test_data_fixed_fp_any = {key: [] for key in get_keys()}

    # gather information
    for folder in folders:
        if folder.endswith("xcl"):
            continue

        sep_loc = [m.start() for m in re.finditer("_", folder)]
        fine_desc = folder[sep_loc[3] + 1:sep_loc[-5] if folder.endswith("_OF") else sep_loc[-4]]

        gen_data["model_desc"].append(fine_desc)

        train_info = os.path.join(source_dir, folder, "train_info.json")
        test_info = os.path.join(source_dir, folder, "test_info.json")

        f_train_info = open(train_info, "r")
        f_test_info = open(test_info, "r")

        json_train = json.load(f_train_info)
        json_test = json.load(f_test_info)

        f_train_info.close()
        f_test_info.close()

        for key in get_keys():
            train_data_mid_threshold[key].append(json_train["best_model"]["train_metrics"][key])
            valid_data_mid_threshold[key].append(json_train["best_model"]["validation_metrics"][key])
            test_data_fixed_fp_all[key].append(json_test["fixed-fp-all"]["metric"][key])
            test_data_fixed_fp_any[key].append(json_test["fixed-fp-any"]["metric"][key])

    fig = plt.figure()
    axs = fig.subplots(2, 2)

    sns.swarmplot(
        x=gen_data["model_desc"],
        y=test_data_fixed_fp_any["tp_rate_recall_sens"],
        hue=test_data_fixed_fp_all["tp_rate_recall_sens"],
        legend=False,
        ax=axs[0, 0]
    ).set_title("baseline")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
