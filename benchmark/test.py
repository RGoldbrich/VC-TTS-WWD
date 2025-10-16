# standard lib
import json
import logging
import os
import re

# third party
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch
from torch.utils.data import DataLoader

# application
import common.log as log
from benchmark.visualizations import yield_train_out_folders
from common.audio import build_feature_list, ClassFlags, WORKING_SAMPLING_RATE
from common.metrics import compute_metrics_from_cm
from data.features import SETT
from model.classifier import get_master_model
from model.dataset import EagerDataset


def yield_sublist(orig_list: np.array, sublist_len: int = 3):
    for i in range(orig_list.shape[0] - sublist_len + 1):
        yield orig_list[i:i + sublist_len]


def get_threshold_from_windows(
        windows: list[float],
) -> float:
    threshold = 0

    for three_windows in yield_sublist(windows):
        threshold = max(min(three_windows), threshold)

    return threshold


def get_global_pred_from_windows(
        windows: list[float],
        thresholds: np.array,
) -> dict:
    """
    Computes the number of times an audio is accepted, based on the predictions of overlapping windows.
    :param windows: List of network outputs
    :param thresholds: List of thresholds between 0 and 1 to test against
    :return: dictionary with n_accepts and thresholds for accepts per window (used for auc roc)
    """

    n_rules = thresholds.shape[0]

    # for each window individually
    n_accepts = np.zeros(n_rules, np.int32)
    cooldown = np.zeros(n_rules, np.int32)
    decision_boundary = np.zeros(len(windows) - 2, dtype=np.float32)

    # min three moderated windows
    n_accepts_three_min = np.zeros(n_rules, np.int32)
    cooldown_three_min = np.zeros(n_rules, np.int32)
    decision_boundary_three_min = np.zeros(len(windows) - 2, dtype=np.float32)

    # mean three moderated windows
    n_accepts_three_mean = np.zeros(n_rules, np.int32)
    cooldown_three_mean = np.zeros(n_rules, np.int32)
    decision_boundary_three_mean = np.zeros(len(windows) - 2, dtype=np.float32)

    # convert list to numpy array
    windows = np.asarray(windows)

    # iterate over all continuous sub-lists with length 3
    for idx, three_windows in enumerate(yield_sublist(windows)):
        # decision boundary for individual windows is just this windows prediction
        decision_boundary[idx] = three_windows[0]

        # decision boundary for three windows is the min of all predictions.
        min_prediction = np.min(three_windows)
        decision_boundary_three_min[idx] = min_prediction

        # decision boundary for three windows is the mean of all predictions.
        mean_prediction = np.mean(three_windows)
        decision_boundary_three_mean[idx] = mean_prediction

        # compute if this frame is accepted
        is_accepted = three_windows[0] > thresholds
        is_accepted_three_min = min_prediction > thresholds
        is_accepted_three_mean = mean_prediction > thresholds

        # increase where cooldown is 0 and frame is accepted
        n_accepts[(cooldown == 0) & is_accepted] += 1
        n_accepts_three_min[(cooldown_three_min == 0) & is_accepted_three_min] += 1
        n_accepts_three_mean[(cooldown_three_mean == 0) & is_accepted_three_mean] += 1

        # step cooldown
        cooldown = np.maximum(cooldown - 1, 0)
        cooldown_three_min = np.maximum(cooldown_three_min - 1, 0)
        cooldown_three_mean = np.maximum(cooldown_three_mean - 1, 0)

        # reset cool down if accepted (set to 1, for rising edge detection)
        cooldown[is_accepted] = 0
        cooldown_three_min[is_accepted_three_min] = 0
        cooldown_three_mean[is_accepted_three_mean] = 0

    return {
        "n_accepts": n_accepts,
        "n_accepts_three_min": n_accepts_three_min,
        "n_accepts_three_mean": n_accepts_three_mean,
        "decision_boundary": decision_boundary,
        "decision_boundary_three_min": decision_boundary_three_min,
        "decision_boundary_three_mean": decision_boundary_three_mean,
    }


def find_pivot_given_fp_rate(
        accept_thresholds: list[float],
        expected_labels: list[float],
        fp_target: float,
) -> tuple[dict, float]:
    metric_dict = dict()
    search_pivot = 0.5

    binary_search_bounds = [0.0, 1.0]
    for i in range(20):
        search_pivot = (binary_search_bounds[1] + binary_search_bounds[0]) * .5
        search_predicted_labels = [1 if p > search_pivot else 0 for p in accept_thresholds]

        search_cm = metrics.confusion_matrix(expected_labels, search_predicted_labels, labels=[0, 1]).ravel()
        metric_dict = compute_metrics_from_cm(*search_cm)

        logging.info(f"search_pivot: {search_pivot} with fp: {metric_dict['fp']}")

        if metric_dict["fp"] > fp_target:
            binary_search_bounds[0] = search_pivot
        else:
            binary_search_bounds[1] = search_pivot

    logging.info(f"binary search for false accept rate: {metric_dict}")

    return metric_dict, search_pivot


def predict_precomputed_features(
        cl,
        dirs_label_fraction: list[tuple[str, int]],
        out_dir: str = None,
        label_for_out_dir: str = None,
        extra_label: str = None,
        supress_log: bool = False,
        device: str = "cpu",
) -> (dict, int, str):
    # create log and intermediate data directory
    log_handler = None
    if not supress_log:
        log_file = "/func.log" if out_dir is None else "/auto_test_func.log"

        if not out_dir:
            if extra_label:
                label = label_for_out_dir + "_" + extra_label
            else:
                label = label_for_out_dir + "_Default"
            out_dir = log.setup_out_dir(os.getcwd(), "benchmark", label)

        log_handler = log.setup_logging(out_dir + log_file)
        logging.info("start predict_precomputed_features()")
        logging.info(f"params: {locals()}")

    if not supress_log:
        logging.info(f"device: {device}")

    # move model to device and set to eval mode
    cl.to(device)
    cl.eval()

    # array of decision boundaries to test
    steps = torch.arange(-10, 10.05, 0.05)
    thresholds = torch.nn.functional.sigmoid(steps).numpy()

    down_shift = 2 * thresholds[0] - thresholds[1]
    scaling = 1 / (1 - 2 * down_shift)
    thresholds = (thresholds - down_shift) * scaling

    n_rules = steps.shape[0]

    # keep track of total number of expected accepts and duration of negative samples outside here
    # make this a list of lists with each outside list representing a decision boundary
    true_accepts = np.zeros(n_rules, np.int32)
    true_accepts_three_min = np.zeros(n_rules, np.int32)
    true_accepts_three_mean = np.zeros(n_rules, np.int32)
    total_accepts = 0

    # make this a list of number of false activations per decision boundary
    false_accepts = np.zeros(n_rules, np.int32)
    false_accepts_three_min = np.zeros(n_rules, np.int32)
    false_accepts_three_mean = np.zeros(n_rules, np.int32)
    total_negative_duration_s = 0

    reports = dict()

    # store expected labels and predictions for roc curve
    both_labels_present = 0
    expected_for_roc = np.array([])
    predicted_for_roc = np.array([])
    predicted_for_roc_three_min = np.array([])
    predicted_for_roc_three_mean = np.array([])

    for directory, label in dirs_label_fraction:
        folder = directory.split("/")[-1]

        data_set = EagerDataset([(directory, label)], None, cl.expects_rnn_shaped_features, supress_log)
        data_loader = DataLoader(data_set, 512, False)

        if label == 1 and os.path.exists(os.path.join(directory, "tags.json")):
            if not supress_log:
                logging.info(f"Found tags.json in {folder}")

            with open(os.path.join(directory, "tags.json"), "r") as f:
                tags = json.load(f)

            df = pd.read_csv(os.path.join(directory, "info.csv"))
        else:
            tags = None
            df = pd.DataFrame()

        # get all predictions regardless of alignment
        with torch.no_grad():
            predictions = []
            for idx, (features, _) in enumerate(data_loader):
                features = features.to(device)

                batch_predictions = cl(features)
                batch_predictions = torch.nn.functional.sigmoid(batch_predictions)

                predictions += batch_predictions.squeeze().tolist()

        # load alignment info
        alignment_files = [name for name in os.listdir(directory) if
                           name.endswith(".pt") and name.find("alignment") >= 0]
        # SORT ALIGNMENT FILES!
        alignment_files = sorted(alignment_files)

        alignments = []
        for af in alignment_files:
            path = os.path.join(directory, af)
            alignments.extend(torch.load(path).tolist())

        assert len(predictions) == len(alignments)

        # compute bounds corresponding with original audio files
        alignment_bounds = []
        current_value = alignments[0]
        current_index = 0
        for i in range(1, len(alignments)):
            if alignments[i] != current_value:
                alignment_bounds.append(((current_index, i), current_value))
                current_index = i
                current_value = alignments[i]
        alignment_bounds.append(((current_index, len(alignments)), current_value))

        if label == 1:
            total_accepts += len(alignment_bounds)

            accepts = np.zeros(n_rules, np.int32)
            accepts_three_min = np.zeros(n_rules, np.int32)
            accepts_three_mean = np.zeros(n_rules, np.int32)

            decision_boundary = np.zeros(len(alignment_bounds), dtype=np.float32)
            decision_boundary_three_min = np.zeros(len(alignment_bounds), dtype=np.float32)
            decision_boundary_three_mean = np.zeros(len(alignment_bounds), dtype=np.float32)

            # contains decision boundary and tags (if tags where present)
            db_and_tags = []

            for idx, (alignment_bound, alignment_idx) in enumerate(alignment_bounds):
                res_dict = get_global_pred_from_windows(
                    predictions[alignment_bound[0]:alignment_bound[1]],
                    thresholds)

                # only count a single positive
                accepts += (res_dict["n_accepts"] > 0).astype(np.int32)
                accepts_three_min += (res_dict["n_accepts_three_min"] > 0).astype(np.int32)
                accepts_three_mean += (res_dict["n_accepts_three_mean"] > 0).astype(np.int32)

                # store max predictions for roc computation
                decision_boundary[idx] = np.max(res_dict["decision_boundary"])
                decision_boundary_three_min[idx] = np.max(res_dict["decision_boundary_three_min"])
                decision_boundary_three_mean[idx] = np.max(res_dict["decision_boundary_three_mean"])

                # store info about decision boundary and tags if present
                if tags and not df.empty:
                    # get the underlying source wave using the alignment index and info.csv dataframe
                    filename = df[df["alignment"] == alignment_idx]["filename"].tolist()[0]

                    # get tags of this audio and store threshold and tags in separate list
                    in_file_tags = [ent["tags"] for ent in tags if (ent["file_name"].split('/')[-1] == filename)]
                    if in_file_tags:
                        db_and_tags.append((decision_boundary_three_min[idx], in_file_tags[0]))

            # store threshold and tags list
            db_and_tags_df = pd.DataFrame(db_and_tags, columns=["decision_boundary", "tags"])
            db_and_tags_df.to_csv(os.path.join(out_dir, "db_and_tags.csv"))

            # store report of this feature folder
            reports[folder] = {
                "label": 1,
                "n_total": len(alignment_bounds),
                "accepts": accepts.tolist(),
                "accepts_three_min": accepts_three_min.tolist(),
                "accepts_three_mean": accepts_three_mean.tolist(),
                "accepts_rel": (accepts / len(alignment_bounds)).tolist(),
                "accepts_rel_three_min": (accepts_three_min / len(alignment_bounds)).tolist(),
                "accepts_rel_three_mean": (accepts_three_mean / len(alignment_bounds)).tolist(),
            }

            # increment total number of true accepts for global cm
            true_accepts += accepts
            true_accepts_three_min += accepts_three_min
            true_accepts_three_mean += accepts_three_mean

            # append to global list of predictions for combined roc computation
            both_labels_present = both_labels_present | 1
            expected_for_roc = np.append(expected_for_roc, np.ones(len(alignment_bounds)))
            predicted_for_roc = np.append(predicted_for_roc, decision_boundary)
            predicted_for_roc_three_min = np.append(predicted_for_roc_three_min, decision_boundary_three_min)
            predicted_for_roc_three_mean = np.append(predicted_for_roc_three_mean, decision_boundary_three_mean)

            df["max_pred"] = decision_boundary
            df["max_pred_min"] = decision_boundary_three_min
            df["max_pred_mean"] = decision_boundary_three_mean

            df.to_csv(f"{out_dir}/info.csv", index=False)

        elif label == 0:
            # duration_s = len(predictions) * (SETT["infe_hop_len"] / WORKING_SAMPLING_RATE)
            duration_s = 83509.91
            total_negative_duration_s += duration_s
            res_dict = get_global_pred_from_windows(predictions, thresholds)

            accepts = res_dict["n_accepts"]
            accepts_three_min = res_dict["n_accepts_three_min"]
            accepts_three_mean = res_dict["n_accepts_three_mean"]

            # increment total number of false accepts for global cm
            false_accepts += accepts
            false_accepts_three_min += accepts_three_min
            false_accepts_three_mean += accepts_three_mean

            # store report of this feature folder
            reports[folder] = {
                "label": 0,
                "duration_s": duration_s,

                "accepts": accepts.tolist(),
                "accepts_three_min": accepts_three_min.tolist(),
                "accepts_three_mean": accepts_three_mean.tolist(),

                "accepts_rel": (accepts / duration_s * 3600).tolist(),
                "accepts_rel_three_min": (accepts_three_min / duration_s * 3600).tolist(),
                "accepts_rel_three_mean": (accepts_three_mean / duration_s * 3600).tolist(),
            }

            both_labels_present = both_labels_present | 2
            expected_for_roc = np.append(expected_for_roc, np.zeros(len(predictions) - 2))
            predicted_for_roc = np.append(predicted_for_roc, res_dict["decision_boundary"])
            predicted_for_roc_three_min = np.append(predicted_for_roc_three_min,
                                                    res_dict["decision_boundary_three_min"])
            predicted_for_roc_three_mean = np.append(predicted_for_roc_three_mean,
                                                     res_dict["decision_boundary_three_mean"])

    if total_accepts > 0:
        reports["combined_true_accepts"] = {
            "label": 1,
            "n_total": total_accepts,

            "accepts": true_accepts.tolist(),
            "accepts_three_min": true_accepts_three_min.tolist(),
            "accepts_three_mean": true_accepts_three_mean.tolist(),

            "accepts_rel": (true_accepts / total_accepts).tolist(),
            "accepts_rel_three_min": (true_accepts_three_min / total_accepts).tolist(),
            "accepts_rel_three_mean": (true_accepts_three_mean / total_accepts).tolist()
        }

    fixed_fp_idx = n_rules // 2

    if total_negative_duration_s > 0:
        reports["combined_false_accepts"] = {
            "label": 0,
            "duration_s": total_negative_duration_s,

            "accepts": false_accepts.tolist(),
            "accepts_three_min": false_accepts_three_min.tolist(),
            "accepts_three_mean": false_accepts_three_mean.tolist(),

            "accepts_rel": (false_accepts / total_negative_duration_s * 3600).tolist(),
            "accepts_rel_three_min": (false_accepts_three_min / total_negative_duration_s * 3600).tolist(),
            "accepts_rel_three_mean": (false_accepts_three_mean / total_negative_duration_s * 3600).tolist(),
        }

        false_accepts_three_min_per_hour = false_accepts_three_min / total_negative_duration_s * 3600
        arg_max = np.argmax(false_accepts_three_min_per_hour)
        false_accepts_three_min_per_hour = false_accepts_three_min_per_hour.tolist()

        # find the lowest db resulting in less than or equal to .5 false accept per hour
        for i in range(n_rules):
            if false_accepts_three_min_per_hour[i] <= 0.5 and i >= arg_max:
                fixed_fp_idx = i
                break

    if not supress_log:
        logging.info("=============================")
        logging.info(f"threshold: {thresholds[fixed_fp_idx]}")
        for key in reports:
            logging.info(f"{key}")
            if reports[key]["label"] == 1:
                logging.info(f"  {reports[key]['accepts_rel_three_min'][fixed_fp_idx]:.2%} tp")
            else:
                logging.info(f"  {reports[key]['accepts_rel_three_min'][fixed_fp_idx]:.4} fp per hour")

    reports["thresholds"] = thresholds.tolist()
    reports["aux"] = {"fixed_fp_threshold": str(thresholds[fixed_fp_idx])}

    # area under the receiver operating characteristic curve
    if both_labels_present == 3:
        auc_roc = metrics.roc_auc_score(expected_for_roc, predicted_for_roc)
        auc_roc_three_min = metrics.roc_auc_score(expected_for_roc, predicted_for_roc_three_min)
        auc_roc_three_mean = metrics.roc_auc_score(expected_for_roc, predicted_for_roc_three_mean)

        reports["aux"]["auc_roc"] = str(auc_roc)
        reports["aux"]["auc_roc_three_min"] = str(auc_roc_three_min)
        reports["aux"]["auc_roc_three_mean"] = str(auc_roc_three_mean)

    # save test info
    if not supress_log:
        with open(f"{out_dir}/test_info.json", mode="w", encoding='utf-8') as f:
            json.dump(reports, f)

        np.save(f"{out_dir}/expected.npy", expected_for_roc)
        np.save(f"{out_dir}/predicted_raw.npy", predicted_for_roc)
        np.save(f"{out_dir}/predicted_min.npy", predicted_for_roc_three_min)
        np.save(f"{out_dir}/predicted_mean.npy", predicted_for_roc_three_mean)

    if log_handler:
        log.close_logging(log_handler)

    updated_out_dir = out_dir + f"_{str(thresholds[fixed_fp_idx])[2:6]}"
    if "combined_true_accepts" in reports:
        updated_out_dir += f"_{str(reports['combined_true_accepts']['accepts_rel_three_min'][fixed_fp_idx])[2:5]}tp"

    if "combined_false_accepts" in reports:
        updated_out_dir += f"_{str(reports['combined_false_accepts']['accepts_rel_three_min'][fixed_fp_idx])[2:5]}fp"

    if not supress_log:
        os.rename(out_dir, updated_out_dir)
        logging.info("end of predict_precomputed_features")

    return reports, fixed_fp_idx, updated_out_dir
