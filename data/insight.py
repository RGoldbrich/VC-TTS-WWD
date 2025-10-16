# standard lib
import csv
import json
import logging
import os
from time import time

# third party
import numpy as np
import pandas as pd
import pydub
import pydub.playback
import sounddevice  # this is required to suppress ALSA LIB warnings
import torch
import torchaudio
import webrtcvad

# application
import common.audio as audioc
import common.log as log

# global
vad = webrtcvad.Vad(3)


# note: DO NOT name this file inspect.py


def get_webrtcvad_segments(
        wave: torch.Tensor,
        sr: int,
) -> tuple[np.array, int]:
    # resample to match working sampling rate
    if sr != 16000:
        wave = torchaudio.functional.resample(wave, sr, 16000)

    # convert from 32-bit float normalized in [-1, 1] to 16-bit int
    wave_16int = (wave * 32767).to(torch.int16)

    # must be either 10, 20 or 30
    # this "segment" is refereed to as "frame" in webrtcvad documentation!
    segment_duration = 20

    # frames per segment to process
    frames_per_segment = int(16000 * segment_duration / 1000)

    arr_is_speech = []

    # subtract frames_per_segment from boundary since webrtcvad can only process full 10/20/30 ms segments
    for start in range(0, len(wave_16int) - frames_per_segment, frames_per_segment):
        end = start + frames_per_segment
        frame = wave_16int[start:end].numpy().tobytes()
        arr_is_speech.append(vad.is_speech(frame, 16000, ))

    return np.array(arr_is_speech), segment_duration


def find_speech_bounds(wave: torch.Tensor, sr: int) -> tuple[int, int]:
    try:
        vad_segs, seg_duration_ms = get_webrtcvad_segments(wave, sr)
    except Exception as e:
        return 0, len(wave)  # return whole wave

    # get first and last segment containing speech
    positive_segs = np.nonzero(vad_segs)[0]
    if len(positive_segs) == 0:
        return 0, len(wave)  # return whole wave

    # speech begin and end in frames
    begin = int(positive_segs[0] * (seg_duration_ms / 1000) * sr)
    end = min(int((positive_segs[-1] + 1) * (seg_duration_ms / 1000) * sr), len(wave))
    return begin, end


def play_least_confident(source_dir: str):
    df = pd.read_csv(os.path.join(source_dir, 'info.csv'))

    print("shape before", df.shape[0])
    df = df[df['confidence'] > 0.25]
    print("shape before", df.shape[0])

    df.sort_values(by="confidence", inplace=True, ascending=True)
    for index, row in df.iterrows():
        print(row)
        audioc.play_audio_file(os.path.join(source_dir, row['prepare_filename']))


def light_inspect(source_dir: str, run_vad: bool, out_label: str):
    out_dir = log.setup_out_dir(os.getcwd(), "inspect", out_label)
    log_handler = log.setup_logging(out_dir + "/func.log")
    logging.info("start of inspect_dataset")
    logging.info(f"light_max_volume() params: {locals()}")

    files = audioc.get_audio_files_by_dir(source_dir)
    n_files = len(files)
    logging.info(f"{n_files} audio files in {source_dir} found")

    raw_len_s = 0

    vad_errors, all_negative_errors = 0, 0

    info = []
    ts_begin = time()

    for index, file in enumerate(files):
        try:
            wave, sr = torchaudio.load(file)
        except Exception as e:
            logging.warning(f"unable to open {file}: {e}")
            continue

        wave = audioc.to_mono_drop_ch_dim(wave)
        raw_len_s += len(wave) / sr

        if run_vad:
            try:
                vad_segs, seg_duration_ms = get_webrtcvad_segments(wave, sr)
            except Exception as e:
                logging.warning(f"unable process {file}: {e}")
                vad_errors += 1
                continue

            # get first and last segment containing speech
            positive_segs = np.nonzero(vad_segs)[0]
            if len(positive_segs) == 0:
                all_negative_errors += 1
                continue

            # speech begin and end in frames
            begin = int(positive_segs[0] * (seg_duration_ms / 1000) * sr)
            end = min(int((positive_segs[-1] + 1) * (seg_duration_ms / 1000) * sr), len(wave))
            duration_s = (end - begin) / sr
        else:
            # consider whole audio as content region
            begin = 0
            end = len(wave)
            duration_s = 0  # set to zero to indicate no vad run

        volume_db = audioc.wave_to_db(wave[begin:end])

        max_db = torch.max(volume_db).item()
        mean_db = torch.mean(volume_db).item()

        info.append((
            file.split('/')[-1],
            duration_s,
            max_db,
            mean_db,
        ))

        log.log_time(ts_begin, index, n_files, 100)

    logging.info(f"number of files: {n_files}")
    logging.info(f"raw duration [s]: {raw_len_s:.2f} {raw_len_s / 60:.0f} {raw_len_s / 3600:.1f}")
    logging.info(f"avg duration [s]: {raw_len_s / n_files:.2f}")
    df = pd.DataFrame(info, columns=["filename", "duration_s", "vol_max_db", "vol_mean_db"])

    df.to_csv(os.path.join(out_dir, "info.csv"), index=False)


def compute_volume_inplace(source_dir: str):
    df = pd.read_csv(os.path.join(source_dir, "info.csv"))

    col = "preprocess_filename" if "preprocess_filename" in df.columns else "prepare_filename"

    info = []

    for index, row in df.iterrows():
        try:
            wave, sr = torchaudio.load(os.path.join(source_dir, row[col]))
        except Exception as e:
            print(e)
            continue

        max_db = audioc.max_db(wave)
        mean_db = audioc.mean_db(wave)

        info.append((
            row["filename"],
            max_db,
            mean_db
        ))

    volume_df = pd.DataFrame(info, columns=["filename", "vol_max_db", "vol_mean_db"])

    df = pd.merge(df, volume_df, on="filename", suffixes=("_leg", ""))
    df.to_csv(os.path.join(source_dir, "info.csv"), index=False)
    print("done")
