# standard lib
import json
import os
import random
import shutil

# third party
import pandas as pd
import torchaudio

# application
import common.audio as audioc
from data.insight import get_webrtcvad_segments


def copy_sonos(json_path: str, target_dir: str, hotword_label: int = 1) -> None:
    json_dir = os.path.dirname(json_path)

    n_hotwords = 0

    with open(json_path) as js_file:
        # js is array of elements containing: "duration", "worker_id", "audio_file_path", "id" and "is_hotword"
        js = json.load(js_file)

        # for every element in json array
        for entry in js:
            # if wake word
            if entry["is_hotword"] is hotword_label:
                # copy to target directory
                shutil.copy(
                    os.path.join(json_dir, entry["audio_file_path"]),
                    target_dir)

                n_hotwords += 1

    print(f"{n_hotwords} elements copied!")


def copy_dipco(source_dir: str, target_dir: str) -> None:
    files = os.listdir(source_dir)

    n_copied = 0
    for file in files:
        if file.endswith(".CH7.wav"):
            print(f"{n_copied} - {file}")
            shutil.copy(os.path.join(source_dir, file),
                        os.path.join(target_dir, file))
            n_copied += 1

    print(f"{n_copied} elements copied!")


def split_musan(source_dir: str, train_dir: str, split: float = 0.75) -> None:
    files = audioc.get_audio_files_by_dir(source_dir)
    random.shuffle(files)

    split_index = int(len(files) * split)
    train_files = files[:split_index]

    for file in train_files:
        relative_path = os.path.join(*file.split('/')[-3:])
        destination_path = os.path.join(train_dir, relative_path)

        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        shutil.move(file, destination_path)
        # print(f"moving {file} -> {os.path.join(train_dir, '/'.join(file.split('/')[-3:]))}")


def extract_ref_ds(source_dir: str, target_dir: str, length_bounds=(4, 15), n_files=5000) -> None:
    files = sorted(audioc.get_audio_files_by_dir(source_dir))
    print(f"number of files", len(files))
    random.shuffle(files)

    copied = 0
    for idx, file in enumerate(files):
        # VCTK _ mic 1 only
        if "mic1" not in file:
            continue

        try:
            wave, sr = torchaudio.load(file)
        except Exception as e:
            print(e)
            continue
        length = wave.shape[-1] / sr

        if length_bounds[0] <= length <= length_bounds[1]:
            # print(f"copying {file} to {os.path.join(target_dir, file.split('/')[-1])}")
            shutil.copy(file, os.path.join(target_dir, file.split('/')[-1]))
            copied += 1

            if copied >= n_files:
                break

        if idx % 100 == 0:
            print(idx, str(round(copied / (idx + 1) * 100, 1)) + "%")


def mcv_to_wav(source_dir: str, target_dir: str, length_bounds=(7.5, 15), n_files=5000) -> None:
    files = sorted(audioc.get_audio_files_by_dir(source_dir))
    print(f"number of files", len(files))
    random.shuffle(files)

    copied = 0
    for file in files:
        try:
            wave, sr = torchaudio.load(file)
        except Exception as e:
            continue

        length = wave.shape[-1] / sr

        if length_bounds[0] <= length <= length_bounds[1]:
            print(f"copying {file} to {os.path.join(target_dir, file.split('/')[-1])}")
            torchaudio.save(os.path.join(target_dir, file.split('/')[-1][:-3] + "wav"), wave, sr)
            copied += 1

            if copied >= n_files:
                break

    print("copied", copied)


def split_peoplespeech_by_vad(source_dir: str, length_bounds=(10, 15)) -> None:
    files = sorted(audioc.get_audio_files_by_dir(source_dir))

    good_dir = "../Datasets/speech_voice/People_Speech_Valid_Good_Selection"
    bad_dir = "../Datasets/speech_voice/People_Speech_Valid_Bad_Selection"

    copied = [0, 0]
    for idx, file in enumerate(files):
        wave, sr = torchaudio.load(file)
        length = wave.shape[-1] / sr

        if length < length_bounds[0] or length > length_bounds[1]:
            continue

        wave = audioc.to_mono_drop_ch_dim(wave)

        segments, _ = get_webrtcvad_segments(wave, sr)

        valid = not any(segments[-50:]) and not any(segments[:20]) and any(segments)

        if valid:
            out_file_name = os.path.join(good_dir, file.split('/')[-1])
            copied[1] += 1
        else:
            out_file_name = os.path.join(bad_dir, file.split('/')[-1])
            copied[0] += 1

        shutil.copy(file, out_file_name)

        if idx % 50 == 0:
            print(f"{copied[0]}/{copied[1]}")

        if copied[1] >= 100:
            break


def create_info_csv():
    txt_file = "../Projects/SSD-KWS/data/transcripts.txt"

    with open(txt_file, "r") as file:
        lines = [l.strip() for l in file.readlines()]

    filenames = [f"{i}.wav" for i in range(5000)]

    dictionary = {
        "filename": filenames,
        "transcript": lines[30000:35000],
    }

    dest = "../Datasets/synthetic_data/TTS_Raw_VCTK/info.csv"
    pd.DataFrame(dictionary).to_csv(dest, index=False)


def extract_tedlium(source_dir: str, target_dir: str, target_length=12.5) -> None:
    # extract three in different location throughout the talk?
    files = sorted(audioc.get_audio_files_by_dir(source_dir))
    print(f"number of files", len(files))
    random.shuffle(files)

    copied = 0
    for file in files:
        wave, sr = torchaudio.load(file)
        length = wave.shape[-1] / sr

        if length >= 60:
            for i in range(2):
                start = (i + 1) * 0.333 * length

                torchaudio.save(os.path.join(target_dir, file.split('/')[-1][:-4] + f"{i}.wav"),
                                wave[:, int(start * sr):int((start + target_length) * sr)], sr)

                copied += 1

    print("copied", copied)
