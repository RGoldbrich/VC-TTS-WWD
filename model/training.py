# standard lib
import copy
import csv
import gc
import json
import logging
import numbers
import os
from time import time, sleep

# third party
import sklearn.metrics as metrics
import torch.cuda
import torchinfo
from torch._C._jit_tree_views import FalseLiteral
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# application
import common.log as log
from benchmark.test import predict_precomputed_features
from common.audio import build_feature_list, ClassFlags, FeatDesc
from model.classifier import get_master_model
from model.dataset import EagerDataset
from model.visualizations import compute_metrics_from_cm


def train(
        cl,
        train_feat: list[tuple[str, int]],
        train_filter: list[tuple[float, float, tuple[float, float], float]],
        test_feat: list[tuple[str, int]],
        feature_stub: str,
        epochs: int = 100,
        valid_split_perc: float = 0.1,
        lr: float = 0.001,
        pos_class_weight_factor: float = 1,
        batch_size: int = 128,
        num_workers: int = 0,
        extra_label: str = None,
) -> None:
    if extra_label:
        label = cl.architecture_stub + "_" + feature_stub + "_" + extra_label
    else:
        label = cl.architecture_stub + "_" + feature_stub + "_Default"

    # create log and intermediate data directory
    out_dir = log.setup_out_dir(os.getcwd(), "train", label)
    log_handler = log.setup_logging(out_dir + "/func.log")
    logging.info("start train()")
    logging.info(f"params: {locals()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"device: {device}")

    logging.info(torchinfo.summary(cl, cl.expected_feature_size, device=device, verbose=0))

    # move model to device
    cl.to(device)

    # load dataset
    logging.info(f"loading dataset")
    ds = EagerDataset(train_feat, train_filter, cl.expects_rnn_shaped_features)
    logging.info("loading dataset done")

    # train test split
    valid_size = int(valid_split_perc * len(ds))
    train_size = len(ds) - valid_size
    train_size_deci = int(train_size * 0.1)
    train_set, valid_set = torch.utils.data.random_split(ds, [train_size, valid_size])
    train_set_deci, _ = torch.utils.data.random_split(train_set, [train_size_deci, len(train_set) - train_size_deci])

    # recognition test
    test_dirs_label_fraction_pos_only = [t for t in test_feat if t[1]]

    # dataloaders
    train_loader = DataLoader(train_set, batch_size, True, pin_memory=(device == "cuda"), num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size, True, pin_memory=(device == "cuda"), num_workers=num_workers)
    train_deci_loader = DataLoader(train_set_deci, batch_size, True, pin_memory=(device == "cuda"),
                                   num_workers=num_workers)

    # label weights
    label_weights = ds.get_label_weights()
    pos_class_weight = label_weights[0] / label_weights[1]
    logging.info(f"{label_weights[0]} neg / {label_weights[1]} pos (~{pos_class_weight:.1f}:1)")
    weight = torch.Tensor([pos_class_weight * pos_class_weight_factor]).to(device)

    # batch_loss and optimizer
    loss_function = BCEWithLogitsLoss(weight)
    optimizer = AdamW(cl.parameters(), lr)

    # stop training after no improvement following FOUR learn rate reductions to 1% of the initial lr
    lr_schedule = {"factor": 0.31622776601683794, "patience": 3, "cooldown": 0, }
    lr_stop_condition = lr * (lr_schedule["factor"] ** 4.5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                  factor=lr_schedule["factor"],
                                  patience=lr_schedule["patience"],
                                  cooldown=lr_schedule["cooldown"],
                                  min_lr=1e-10)

    logging.info(f"training on {len(train_set)}")
    logging.info(f"validation on {len(valid_set)}")

    # saving best model performance
    best_model_state = dict()
    best_model = {
        "validation_loss": 1_000_000,
        "validation_metrics": dict(),
        "epoch": 0
    }

    # keep track of previous learn rate to print info at beginning of epoch
    previous_learn_rate = optimizer.param_groups[0]['lr']

    info = []
    info_json = dict()
    ts_begin = time()

    # main training loop
    for e in range(epochs):
        logging.info("========================================================")

        # notify about learn rate reduction
        if optimizer.param_groups[0]['lr'] < previous_learn_rate:
            logging.info(f"LEARN RATE REDUCED")
        previous_learn_rate = optimizer.param_groups[0]['lr']

        logging.info(f"EPOCH {e:3d}, lr={round(optimizer.param_groups[0]['lr'], 8)}")

        # turn on gradient tracking
        cl.train()

        epoch_loss = 0

        progress_print_at = .1

        for idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)

            # convert labels from [batch_size] to [batch_size, 1] to match model prediction
            labels = labels.unsqueeze(1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            predictions = cl.forward(features)

            # compute batch_loss and gradients
            batch_loss = loss_function.forward(predictions, labels)
            batch_loss.backward()

            # clip gradients
            torch.nn.utils.clip_grad_norm_(cl.parameters(), 1.0)

            # update weights
            optimizer.step()

            # add to total epoch loss
            epoch_loss += batch_loss.item()

            # print periodically
            if (idx + 1) * batch_size >= progress_print_at * len(train_set):
                logging.info(f"  {progress_print_at:4.0%}: batch_loss {epoch_loss / (idx * batch_size):.6f}")
                progress_print_at += .1

        # mean epoch loss
        epoch_mean_loss = epoch_loss / len(train_set)

        # validate
        epoch_mean_validation_loss, validation_metrics = validate(cl, valid_loader, device, loss_function)
        epoch_mean_validation_loss /= len(valid_set)

        # reporting fitness
        logging.info("LOSS")
        logging.info(f"  train:      {epoch_mean_loss * 1000:.6f} (/1000)")
        logging.info(f"  valid:      {epoch_mean_validation_loss * 1000:.6f} (/1000)")
        log_metrics("VALIDATION", validation_metrics)

        # stepping lr scheduling
        scheduler.step(epoch_mean_validation_loss)

        # save best model or print delta to best model
        logging.info("BEST")
        if epoch_mean_validation_loss < best_model["validation_loss"]:
            v_loss_reduction = (best_model['validation_loss'] - epoch_mean_validation_loss) / best_model[
                'validation_loss']
            logging.info(f"  new best - {v_loss_reduction:.2%} v_loss reduction")

            best_model["validation_loss"] = epoch_mean_validation_loss
            best_model["epoch"] = e
            best_model["state_path"] = out_dir + f"/{log.get_timestamp_str()}_E{e}"
            best_model["validation_metrics"] = validation_metrics
            best_model_state = copy.deepcopy(cl.state_dict())
        else:
            v_loss_increase = (epoch_mean_validation_loss - best_model['validation_loss']) / best_model[
                'validation_loss']
            logging.info(f"  epoch {best_model['epoch']} ({e - best_model['epoch']} ago) "
                         f"- {v_loss_increase:.2%} v_loss increase")

        # info for epoch including learn-rate, loss and validation confusion matrix
        info.append((
            e, previous_learn_rate, epoch_mean_loss,
            epoch_mean_validation_loss, best_model["epoch"] == e, best_model["validation_loss"],
            validation_metrics["tn"], validation_metrics["fp"], validation_metrics["fn"], validation_metrics["tp"],
        ))

        # report current pace
        pace = int((time() - ts_begin) / (e + 1))
        logging.info("PACE")
        logging.info(f"  {pace:.1f} s/epoch; expected time to go: {int(pace * (40 - e - 1) / 60)} min")

        # stop training if lr falls below threshold
        if optimizer.param_groups[0]['lr'] < lr_stop_condition:
            logging.info(f"Training stopped due to learn-rate falling under threshold of {lr_stop_condition}")
            break

    logging.info("========================================================")
    logging.info(f"BEST MODEL")
    logging.info(f"  {best_model['state_path']}")

    # print best performing model statistics
    log_metrics("ON VALIDATION DATA", best_model["validation_metrics"])

    # run training data through best model
    cl.load_state_dict(best_model_state)
    training_loss, training_metric_dict = validate(cl, train_loader, device, loss_function)
    log_metrics("ON TRAINING DATA", training_metric_dict)

    # save epoch info
    with open(f"{out_dir}/info.csv", mode="w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow((
            "episode", "lr", "epoch_mean_loss",
            "epoch_mean_validation_loss", "is_best_validation_loss_so_far", "best_validation_loss",
            # "train_tn", "train_fp", "train_fn", "train_tp",
            "valid_tn", "valid_fp", "valid_fn", "valid_tp",
            # "test_fn", "test_tp"
        ))
        writer.writerows(info)

    overfitting_factor = training_metric_dict["f1_score"] / best_model["validation_metrics"]["f1_score"]
    if overfitting_factor <= 1.012:  # allows about 1% deviation assuming f1 score > 90%
        logging.info(f"  Overfitting factor {overfitting_factor:.5f}")
    else:
        logging.warning(f"HIGH OVERFITTING FACTOR OF {overfitting_factor:.5f}")

    info_json["best_model"] = best_model
    info_json["best_model"]["train_metrics"] = training_metric_dict
    info_json["best_model"]["of_factor"] = overfitting_factor

    # save train info
    with open(f"{out_dir}/train_info.json", mode="w", encoding='utf-8') as file:
        json.dump(info_json, file)
    torch.save(best_model_state, info_json["best_model"]["state_path"])

    # free up memory before running tests
    del train_loader, valid_loader, train_deci_loader
    del train_set, valid_set, train_set_deci
    del ds
    gc.collect()
    torch.cuda.empty_cache()

    # run test
    _, _, out_dir = predict_precomputed_features(cl, test_feat, out_dir, device=device)

    if overfitting_factor > 1.012:
        os.rename(out_dir, out_dir + "_OF")

    logging.info("end of train")
    log.close_logging(log_handler)


def log_metrics(header: str, metric_dict: dict, accepts_only: bool = False) -> None:
    logging.info(header)
    logging.info(f"  {metric_dict['tp']:8d} tp {metric_dict['fn']:8d} fn")
    if not accepts_only:
        logging.info(f"  {metric_dict['fp']:8d} fp {metric_dict['tn']:8d} tn")
    logging.info(f"  {metric_dict['tp_rate_recall_sens']:7.3%} tp" if isinstance(
        metric_dict['tp_rate_recall_sens'],
        numbers.Number) else "     nan% tp")
    if not accepts_only:
        logging.info(f"  {metric_dict['tn_rate_spec']:7.3%} tn" if isinstance(
            metric_dict['tn_rate_spec'],
            numbers.Number) else "     nan% tn")
        logging.info(f"  {metric_dict['f1_score']:7.3%} f1" if isinstance(
            metric_dict['f1_score'],
            numbers.Number) else "     nan% f1")


def validate(
        model,
        data_loader,
        device,
        loss_function,
) -> tuple[float, dict]:
    # set to eval mode and disable gradient tracking
    model.eval()
    with torch.no_grad():
        loss = 0
        expected_labels, predicted_labels = [], []

        for idx, (features, labels) in enumerate(data_loader):
            features, labels = features.to(device), labels.to(device)

            labels = labels.unsqueeze(1)

            predictions = model(features)

            loss += loss_function(predictions, labels).item()

            predictions = torch.nn.functional.sigmoid(predictions)  # manual sigmoid activation after loss

            predicted_labels += torch.round(predictions).to("cpu").tolist()
            expected_labels += labels.to("cpu").tolist()

        # compute mean loss
        mean_loss = loss / len(data_loader)

        # compute confusion matrix and metrics
        cm = metrics.confusion_matrix(expected_labels, predicted_labels, labels=[0, 1]).ravel()
        metric_dict = compute_metrics_from_cm(*cm)

    return mean_loss, metric_dict
