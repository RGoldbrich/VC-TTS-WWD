# standard lib
import csv
import json
import logging
import math
import multiprocessing as mp
import os
import shutil
import zlib
from time import time

# third party
import pandas as pd
import torch
import torchaudio

# application
import common.log as log
from common.audio import WORKING_SAMPLING_RATE, WORKING_N_FRAMES, get_audio_files_by_dir

SETT = {
    "infe_win_len": 38400,
    "infe_hop_len": 4800,
    "feat_win_len": int(WORKING_SAMPLING_RATE * 0.025),  # 400 frames, 25ms
    "feat_hop_len": int(WORKING_SAMPLING_RATE * 0.01)  # 160 frames, 10ms
}

SETT["feat_hops_per_infe_hop"] = SETT["infe_hop_len"] // SETT["feat_hop_len"]  # 15  = 4800  / 320
SETT["feat_hops_per_infe_win"] = SETT["infe_win_len"] // SETT["feat_hop_len"]  # 120 = 38400 / 320

SETT["feat_padding"] = SETT["feat_hops_per_infe_win"] - SETT["feat_hops_per_infe_hop"]  # 105 = 120 - 15
SETT["infe_padding"] = SETT["infe_win_len"] - SETT["infe_hop_len"]  # 33600 = 38400 - 4800

# should either be same as window length or power of 2 for computational purposes
N_FFT = SETT["feat_win_len"]

# these two seem to be a reasonable starting point
N_MELS = 21

# spec augment parameters
specaugment_params = {
    13: (2, 2),  # two masks of length 0 or 1 | avg. 7.7%
    21: (3, 2),  # three masks of length 0 or 1 | avg. 7.1%
    33: (4, 2),  # four masks of length 0 or 1 | avg. 6%

    25: (2, 3),  # two masks of length 0, 1, or 2 | avg. 8%
    40: (2, 4),  # two masks of length 0, 1, 2 or 3 | avg. 7.5%
    64: (3, 4),  # three masks of length 0, 1, 2 or 3 | avg. 7.0%
    "time": (3, 5)  # 6 masks of length [0, 6[ | avg. 7.9
}


# custom implementation of log mel filter bank energies transformation since torchaudio does not provide it directly
class LFBE(torch.nn.Module):
    def __init__(self, n_mels):
        super().__init__()
        self.mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(WORKING_SAMPLING_RATE,
                                                                              n_fft=N_FFT,
                                                                              win_length=SETT["feat_win_len"],
                                                                              hop_length=SETT["feat_hop_len"],
                                                                              n_mels=n_mels)

    def forward(self, w):
        mel_specgram = self.mel_spectrogram_transform(w)
        return torch.log(mel_specgram + 1e-6)


def get_lfbe_transform(n_mels):
    return LFBE(n_mels)


def extract_feature(
        feature_type: str,
        files: list[str],
        global_start_idx: int,
        target_size: torch.Size,
        training_data: bool,
        out_dir: str,
        feature_level_info_list: list,
        file_alignment_list: list,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.print_log(f"{mp.current_process().name} using {device}")

    log.print_log(f"{mp.current_process().name} start {global_start_idx}")
    log.print_log(f"{mp.current_process().name} files {len(files)}")

    if feature_type == "lfbe":
        feature_transform = get_lfbe_transform(N_MELS).to(device)
        specaugment_transform = torchaudio.transforms.SpecAugment(*specaugment_params["time"],
                                                                  *specaugment_params[N_MELS])
    else:
        raise Exception('Unable to determine feature type. Must be "mfcc" or "lfbe"')

    ts_begin = time()

    # to be stored directly from this function
    features = []

    # to be returned in main function
    feature_level_info = []  # storing: [filename, idx, chunk, success or error code] // per chunk
    file_alignment = []  # storing: [filename, alignment] // per file

    for idx, file in enumerate(files):
        wave, sr = torchaudio.load(file)

        # ensure all audio files are 16kHz and 2.4s long if not padding wave
        if (wave.shape[1] != WORKING_N_FRAMES and training_data) or sr != WORKING_SAMPLING_RATE:
            feature_level_info.append((file.split('/')[-1], idx + global_start_idx, 0, 0, "wave_shape_or_sr_error"))
            log.print_log(f"{mp.current_process().name}: wave_shape_or_sr_error")
            continue

        n_chunks = math.floor((wave.shape[1] - SETT["infe_win_len"]) / SETT["infe_hop_len"]) + 1

        start_idx = -3 if not training_data else 0
        end_idx = n_chunks + 4 if not training_data else n_chunks

        for n in range(start_idx, end_idx, 1):
            beg = max(0, SETT["infe_hop_len"] * n)
            end = min(wave.shape[1], SETT["infe_hop_len"] * n + SETT["infe_win_len"])
            if end - beg < SETT["feat_win_len"]:
                continue

            wave_projection = wave[:, beg:end]

            # extract feature
            try:
                feature = feature_transform(wave_projection.to("cuda"))

            except Exception:
                feature_level_info.append((file.split('/')[-1], idx + global_start_idx, n, 0, "failed_to_extract_mfcc"))
                log.print_log(f"{mp.current_process().name}: failed_to_extract_mfcc")
                continue

            if not training_data:
                pad_front = max(0, -n) * SETT["feat_hops_per_infe_hop"]
                pad_back = target_size[2] - (feature.shape[2] + pad_front)

                feature = torch.nn.functional.pad(feature, (pad_front, pad_back))

            # ensure correct shape
            if feature.shape != target_size:
                feature_level_info.append((file.split('/')[-1], idx + global_start_idx, n, 0, "shape_does_not_match"))
                log.print_log(f"{mp.current_process().name}: shape_does_not_match ({feature.shape})")
                continue

            # ensure no inf or nan values
            if torch.isinf(feature).any() or torch.isnan(feature).any():
                feature_level_info.append(
                    (file.split('/')[-1], idx + global_start_idx, n, 0, "contains_inf_or_nan_values"))
                log.print_log(f"{mp.current_process().name}: contains_inf_or_nan_values")
                continue

            # time and frequency masking (drop information)
            if training_data:
                feature = specaugment_transform(feature)

            # save the first audio files in each process for validation purposes
            if idx < 3:
                torchaudio.save(f"{out_dir}/{idx + global_start_idx:02d}_{n}.wav", wave_projection, sr)

            feature = feature.to("cpu")
            features.append(feature)
            feature_level_info.append((file.split('/')[-1], idx + global_start_idx, n,
                                       zlib.crc32(feature.contiguous().numpy().tobytes()), "successfully_extracted"))

        file_alignment.append((file.split('/')[-1], idx + global_start_idx))

        log.log_time(ts_begin, idx, len(files), 200, True)

    # single alignment identifier per file (shared between feature of this file)
    alignment = []
    for _, al, _, _, code in feature_level_info:
        if code == "successfully_extracted":
            alignment.append(al)

    stacked_features = torch.stack(features)
    feature_filename = f"{out_dir}/stacked_features_{mp.current_process().pid}.pt"
    torch.save(stacked_features, feature_filename)

    alignment_tensor = torch.tensor(alignment, dtype=torch.int32)
    alignment_filename = f"{out_dir}/alignment_{mp.current_process().pid}.pt"
    torch.save(alignment_tensor, alignment_filename)

    feature_level_info_list.extend(feature_level_info)
    file_alignment_list.extend(file_alignment)


def process_audios(
        src_dir: str,
        feature_type: str = "lfbe",
        target_size: torch.Size = torch.Size((1, 21, 241)),
        training_data: bool = True,
        n_processes: int = mp.cpu_count() // 4,  # to keep it quiet for now
) -> None:
    # create info json and store function parameters
    locals_temp = locals().copy()  # needed to avoid circular reference
    info_json = log.get_or_create_parameter_json(src_dir)
    info_json["feature_params"] = locals_temp

    # create log and intermediate data directory
    if feature_type not in ["lfbe"]:
        raise Exception('Unable to determine feature type. Must be "mfcc" or "lfbe"')

    key = "preprocessing_params" if "preprocessing_params" in info_json.keys() else "preparation_params"

    folder_label = info_json[key]["class"] + "_" + info_json[key][
        "nature"] + "_" + info_json[key]["name"] + "_" + (
                       "TR" if training_data else "EV")

    out_dir = log.setup_out_dir(os.getcwd(), "feature", folder_label)
    log_handler = log.setup_logging(out_dir + "/func.log")
    logging.info("start process_audios")
    logging.info(f"process_audios() params: {locals()}")

    logging.info(
        f"feature window: {SETT['feat_win_len']} fr. or {SETT['feat_win_len'] / WORKING_SAMPLING_RATE * 1000:.1f} ms")
    logging.info(
        f"feature hop: {SETT['feat_hop_len']} fr. or {SETT['feat_hop_len'] / WORKING_SAMPLING_RATE * 1000:.1f} ms")

    logging.info(f"spec augment: {specaugment_params}")

    # sort here, so files receive alignment matching their index and probably name
    df = pd.read_csv(os.path.join(src_dir, "info.csv"), index_col=None)
    col = "preprocess_filename" if "preprocess_filename" in df.columns else "prepare_filename"
    files = [f"{src_dir}/{f}" for f in df[col].tolist()]

    n_files = len(files)
    logging.info(f"{n_files} files found")

    start_time = time()

    files_per_proc = n_files // n_processes
    with mp.Manager() as manager:
        feature_level_info_list = manager.list()
        file_alignment_list = manager.list()

        processes = []
        for pid in range(n_processes):
            start_idx = pid * files_per_proc
            end_idx = (pid + 1) * files_per_proc if pid != n_processes - 1 else n_files

            p = mp.Process(target=extract_feature,
                           args=(feature_type,
                                 files[start_idx:end_idx],
                                 start_idx,
                                 target_size,
                                 training_data,
                                 out_dir,
                                 feature_level_info_list,
                                 file_alignment_list))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        feature_level_info = list(feature_level_info_list)
        file_alignment = list(file_alignment_list)

    # write info json about this feature folder
    info_json["feature_params"]["n_samples"] = n_files  # - (wave_err_c + mfcc_err_c + shape_err_c + nan_err_c)
    info_json["feature_params"]["feature"] = feature_type.upper()
    info_json["feature_params"]["is_training"] = training_data
    info_json["feature_params"]["extraction"] = SETT
    info_json["feature_params"]["size"] = target_size
    info_json["feature_params"]["specaugment"] = {
        "time": specaugment_params["time"],
        "freq": specaugment_params[N_MELS],
    }
    log.store_parameter_json(out_dir, info_json)

    # store csv with "per file" alignment info
    compressed_alignment_df = pd.DataFrame(file_alignment, columns=["filename", "alignment"])
    compressed_df = pd.merge(df, compressed_alignment_df, left_on=col, right_on="filename", suffixes=("", "_y"))
    compressed_df = compressed_df.drop(columns=["filename_y"])
    compressed_df.to_csv(f"{out_dir}/info.csv", index=False)

    # store csv with "per feature" alignment info
    full_alignment_df = pd.DataFrame(feature_level_info, columns=["filename", "alignment", "chunk", "checksum", "code"])
    full_df = pd.merge(df, full_alignment_df, left_on=col, right_on="filename", suffixes=("", "_y"), how="right")
    full_df = full_df.drop(columns=["filename_y"])
    full_df.to_csv(f"{out_dir}/full_info.csv", index=False)

    logging.info("end process_audios")
    log.close_logging(log_handler)


def split_into_sep_folders():
    feat_to_move = [
        ("lfbe_25", "l25"),
        ("mfcc_21", "m21"),
        ("mfcc_33", "m33"),
    ]

    base_dir = "../Projects/SSD-KWS/data/out_feature"

    folders = sorted(os.listdir(base_dir))

    for folder in folders:
        if not folder.startswith("240819"):
            continue

        print("Folder", folder)

        for feat in feat_to_move:
            new_folder = folder.replace("lfbe_64", feat[0])
            destination = os.path.join(base_dir, f"ent_{feat[0]}", new_folder)

            print("Destination", destination)
            os.makedirs(destination)

            for file in os.listdir(os.path.join(base_dir, folder)):
                if feat[1] in file:
                    move_from = os.path.join(base_dir, folder, file)
                    move_to = os.path.join(destination, file)
                    print("Move from", move_from)
                    print("Move to", move_to)

                    shutil.move(move_from, move_to)

                if "alignment" in file:
                    copy_from = os.path.join(base_dir, folder, file)
                    copy_to = os.path.join(destination, file)
                    print("Copy from", copy_from)
                    print("Copy to", copy_to)
                shutil.copy(copy_from, copy_to)
