# standard lib
import gc
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from enum import Flag, auto
from typing import List, Tuple, Any

# third party
import pydub
import pydub.playback
import torch
import torchaudio

# application
from common.log import get_or_create_parameter_json

# valid audio file extensions
EXTENSIONS = [".wav", ".flac", ".mp3", ".sph"]

# dataset root directory
DS_DIR = "../Datasets/"

# sonos snips datasets
SONOS_SNIPS_HotWord_Test = DS_DIR + "speech_voice/Snips/pos_test"
SONOS_SNIPS_Speech_Test = DS_DIR + "speech_voice/Snips/neg_test"
SONOS_SNIPS_HotWord_Train = DS_DIR + "speech_voice/Snips/pos_train"
SONOS_SNIPS_Speech_Train = DS_DIR + "speech_voice/Snips/neg_train"

# raw synthetic datasets
TTS_Raw_Libri = DS_DIR + "synthetic_data/TTS_Raw_Libri"
VC_Raw_PeopleSpeech = DS_DIR + "synthetic_data/VC_Raw_PeopleSpeech"

# synthetic datasets


# speech and voice datasets
LIBRI_Speech = DS_DIR + "speech_voice/LibriSpeech"
LIBRI_Speech_S = DS_DIR + "speech_voice/LibriSpeech_Selection"
VCTK_S = DS_DIR + "speech_voice/VCTK_Selection"
PEOPLE_Speech = DS_DIR + "speech_voice/Peoples_Speech_Valid"
PEOPLE_Speech_S = DS_DIR + "speech_voice/Peoples_Speech_Valid_Selection"
MCV_Speech = DS_DIR + "speech_voice/MCV_full"
MCV_Speech_S = DS_DIR + "speech_voice/MCV_full_Selection"

TEDLIUM_S = DS_DIR + "speech_voice/TEDLIUM_release-3_Selection"

# environment and augmentation datasets
MUSAN_Test = DS_DIR + "environment_augment/musan"
MUSAN_Train = DS_DIR + "environment_augment/musan_train"

# room impulse responses
RIR = "rirs/RIRS_NOISES/real_rirs_cleaned"

WORKING_SAMPLING_RATE = 16000
WORKING_N_FRAMES = 38400


def wave_to_db(w: torch.Tensor) -> torch.Tensor:
    """
    Computes the volume of a wave in decibel at any given point.

    The decibel scale is a logarithmic scale and always relative to a reference pressure (or amplitude).

    The reference amplitude used is 1.0,
    meaning a wave with amplitudes between -1.0 and 1.0 will at most show a volume of 0db.

    A db value change by 20 corresponds to an amplitude change by a factor of 10,
    meaning -20db represent amplitudes between -0.1 and 0.1.

    :param w: wave tensor
    :return: volume (tensor) of wave in decibel
    """
    db = torchaudio.functional.amplitude_to_DB(w, 20, 1e-5, 0)
    return db


def mean_db(w: torch.Tensor) -> float:
    """
    Computes the mean volume of a wave in decibel.
    :param w: wave tensor
    :return: mean volume of wave in decibel
    """
    return torch.mean(wave_to_db(w)).item()


def max_db(w: torch.Tensor) -> float:
    """
    Computes the max volume of a wave in decibel.
    :param w: wave tensor
    :return: max volume of wave in decibel
    """
    return torch.max(wave_to_db(w)).item()


def db_diff_to_amp_factor(db_diff: float) -> float:
    return 10 ** (db_diff / 20)


def normalize(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    """
    Normalize a wave so that its amplitudes are between -1.0 and 1.0.
    :param w: wave tensor
    :return: normalized wave tensor
    """
    cur_max_volume = max_db(w)

    gain = -cur_max_volume
    w = torchaudio.functional.gain(w, gain)

    return w, gain


# def normalize_and_gain(w: torch.Tensor, gain: float) -> tuple[torch.Tensor, float]:
#     """
#     Normalizes a wave and applies a gain transformation afterward.
#     :param w: wave tensor
#     :param gain: gain to apply
#     :return:
#     """
#     cur_max_volume = torch.max(wave_to_db(w)).item()
#
#     g = -cur_max_volume + gain
#     w = torchaudio.functional.gain(w, g)
#
#     return w, g


def gain_clip_safe(w: torch.Tensor, gain: float) -> Tuple[torch.Tensor, float]:
    """
    Applies a gain transformation using the specified gain value, but ensures no amplitude clipping occurs.

    Keeps wave tensor on device currently used.
    :param w: wave tensor
    :param gain: target gain to apply
    :return: tuple containing the transformed wave tensor and the actual gain applied
    """
    cur_max_volume = max_db(w)

    # clip gain to -cur_max_volume, gain by -cur_max_volume is the maximum gain allowed (normalizing the audio)
    g = max(gain, -cur_max_volume)
    w = torchaudio.functional.gain(w, g)

    return w, g


def set_max_volume(w: torch.Tensor, cur_vol, tar_vol) -> Tuple[torch.Tensor, float]:
    g = tar_vol - cur_vol
    w = torchaudio.functional.gain(w, g)


# def linked_speed_pitch_transform(
#         w: torch.Tensor,
#         sr: int,
#         speed_range: tuple[float, float] = (.8, 1.2),  # seems reasonably
# ) -> tuple[torch.Tensor, float]:
#     speed_f = random.uniform(*speed_range)
#
#     # clamp assumed sr frequency to multiple of sr / 32
#     assumned_sr = round((sr * speed_f) / (sr // 32)) * (sr // 32)
#
#     # "change" sample rate of wave and resample back to original sample rate to effectively change audio speed
#     # this also changes the pitch of the audio
#     w = torchaudio.functional.resample(w, int(assumned_sr), sr)
#
#     # recompute actually used speed factor
#     used_speed_f = assumned_sr / sr
#
#     return w, used_speed_f
#
#
# def quick_pitch_gain_transform(
#         w: torch.Tensor,
#         sr: int,
#         speed_range: tuple[float, float] = (.8, 1.2),  # seems reasonably
#         normalize_before_gain: bool = True,
#         gain_range: tuple[int, int] = (-20, 0),  # assuming normalization first; will reduce max amp to 10%
#         device: str = "cpu",
# ) -> torch.Tensor:
#     """
#     Performs a resample to change the clips speed and pitch. For performance reasons, the pitch-shift is linked to the
#     speed change and not done independently.
#     Also adjusts the audios volume.
#     :param w:
#     :param sr:
#     :param speed_range:
#     :param normalize_before_gain:
#     :param gain_range:
#     :param device:
#     :return:
#     """
#     logging.info(f"speed_and_pitch_variation() params: {locals()}")
#     speed_f = random.uniform(*speed_range)
#     print("Speed:", speed_f)
#
#     # "change" sample rate of wave and resample back to original sample rate to effectively change audio speed
#     # this also changes the pitch of the audio
#     w = torchaudio.functional.resample(w.to(device), int(sr * speed_f), sr)
#
#     gain = random.uniform(*gain_range)
#     if normalize_before_gain:
#         w = normalize_and_gain(w, gain)
#     else:
#         w = torchaudio.functional.gain(w, gain)
#
#     return w.to("cpu")


def cut_extend_wave_randomly(
        w: torch.Tensor,
        n_target_frames: int
) -> tuple[torch.Tensor, int]:
    """
    Cuts or extents a wave to meet a specified number of frames.
    When wave needs to be cut it will select a random portion of the input wave.
    When wave needs to be extended it will randomly pad the start and end of the input wave.
    :param w: Input wave
    :param n_target_frames: Number of frames to cut or extent to
    :return:
    """
    wave_len = len(w)
    if wave_len > n_target_frames:
        random_center = int(random.uniform(n_target_frames / 2, wave_len - (n_target_frames / 2)))
    else:
        random_center = int(random.uniform(-(n_target_frames / 2) + wave_len, n_target_frames / 2))

    return cut_extend_wave(w, n_target_frames, random_center)


def cut_extend_wave_randomly_keep_section(
        w: torch.Tensor,
        n_target_frames: int,
        section_center: int,
        section_length: int,
        keep_free_front_back: int = 0
) -> tuple[torch.Tensor, int]:
    """
    Cuts or extents a wave to meet a specified number of frames while
    randomly positioning a section of the input wave in the output wave.
    Expects the section to be smaller that the number of target frames
    :param w: Input wave
    :param n_target_frames: Number of frames to cut or extent to
    :param section_center: Center(-frame) of the section in input wave
    :param section_length: Length of the section in frames
    :param keep_free_front_back: Number of frames to keep free of content in the front and back of audio
    :return:
    """
    w = w.to("cpu")

    one_sided_shifting_space = int((n_target_frames - section_length) / 2)

    if one_sided_shifting_space < 0:
        random_center = section_center
        logging.warning(f"cut_extend_wave_randomly_keep_section(): content is wider than target "
                        "keeping original content center, but removing content")
    elif one_sided_shifting_space - keep_free_front_back < 0:
        random_center = section_center
        logging.warning(f"cut_extend_wave_randomly_keep_section(): content + wanted free space is wider than target "
                        "keeping original content center and writing content to free spaces")
    else:
        random_center = section_center + int(random.uniform(
            -one_sided_shifting_space + keep_free_front_back,
            one_sided_shifting_space - keep_free_front_back))

    return cut_extend_wave(w, n_target_frames, random_center)


def cut_extend_wave(
        w: torch.Tensor,
        n_target_frames: int,
        center_frame: int = None
) -> tuple[torch.Tensor, int]:
    """
    Cuts or extents a wave to meet a specified number of frames.
    Allow control over what frame will be centered in the output wave.
    If not specified symmetrically cuts or extent the wave.
    :param w: Input wave
    :param n_target_frames: Number of frames to cut or extent to
    :param center_frame: Frame index in input wave that will be centered in output wave (can be outside of input wave)
    :return:
    """
    w = w.to("cpu")

    # length of wave in frames and default center if not provided
    # to equally cut/extent wave on both ends
    wave_len = len(w)
    center_frame = center_frame or int(wave_len / 2)

    # number of frames to be cut from the front to move input wave center frame onto output wave center frame
    # if negative, frames will be added in the front
    front_cut = int(center_frame - n_target_frames / 2)

    if front_cut > 0:
        w = w[front_cut:]
    elif front_cut < 0:
        front_padding = torch.zeros(-front_cut)
        w = torch.cat((front_padding, w))

    # frame delta at the back
    # if negative, frames will be added in the back
    back_cut = int(wave_len - front_cut - n_target_frames)

    if back_cut > 0:
        w = w[:-back_cut]
    elif back_cut < 0:
        back_padding = torch.zeros(-back_cut)
        w = torch.concat((w, back_padding))

    # return front_cut to allow for adjustments of frame pointers outside this function
    return w, front_cut


def to_mono_drop_ch_dim(w: torch.Tensor) -> torch.Tensor:
    if w.shape[0] > 1:
        # average over channel dimensions
        return torch.mean(w, dim=0)
    else:
        # simply drop dim
        return w[0]


def get_audio_files_by_dir(source_dir: dir) -> list[str | bytes]:
    return [os.path.join(root, file) for root, _, files in os.walk(source_dir) for file in files if
            any(file.lower().endswith(ext) for ext in EXTENSIONS)]


def get_audio_files_containing_stub_from_dir(
        source_dir: dir,
        stub: str,
) -> list[str | bytes]:
    return [os.path.join(root, file) for root, _, files in os.walk(source_dir) for file in files if
            any(file.lower().endswith(ext) for ext in EXTENSIONS) and stub in file]


class ClassFlags(Flag):
    HOT_WORD = auto()

    # NotHotWords individually split
    NOTHOTWORD = auto()
    HEY = auto()
    SNIPS = auto()
    HEY_SPEECH = auto()
    SPEECH_SNIPS = auto()

    SPEECH = auto()


FEAT_FLAG_DEFAULT = {
    "TTS": ClassFlags.HOT_WORD | ClassFlags.SPEECH,
    "VC": ClassFlags.HOT_WORD | ClassFlags.SPEECH,
}

str_to_class = {
    "Speech": ClassFlags.SPEECH,
    "HotWord": ClassFlags.HOT_WORD,
    "NotHotWord": ClassFlags.NOTHOTWORD,
}


def explode_not_hot_word_class(label: str):
    if "HeyWord" in label:
        return ClassFlags.HEY_SPEECH
    elif "WordSnips" in label:
        return ClassFlags.SPEECH_SNIPS
    elif "Hey" in label:
        return ClassFlags.HEY
    elif "Snips" in label:
        return ClassFlags.SNIPS

    raise Exception(f"Unable to explode NotHotWordClass: {label}")


@dataclass
class FeatDesc:
    classes: ClassFlags
    label_filters: list[str] | None
    feature_filters: tuple[float, float, float | tuple[float, float], float] | None
    rir_tr: bool
    vol_tr: bool


def build_feature_list(source_dir: str, listing: dict):
    """
    Builds list of features to pass to training and evaluation function.

    :param source_dir: single parent directory to recursively search through
    :param listing:
        dictionary containing list of tuples for natures in ["HUM", "TTS", "VC"].
        tuple consists of (ClassFlags, label filter, feature filters)
    :return:
    """

    # req. structure like. [(path, label, fraction)]
    train_feat, test_feat, train_filter = [], [], []

    # track [hum, vc, tts] hotword samples for training
    hotword_sample_composition = [0, 0, 0]
    nature_idx = {
        "HUM": 0,
        "VC": 1,
        "TTS": 2,
    }

    expected_feat_win_length = None
    expected_feat_hop_length = None
    expected_feat_type = None
    expected_feat_size = None

    for root, dirs, files in os.walk(source_dir):

        if files and not dirs:
            info_json = get_or_create_parameter_json(root)
            feat_params = info_json["feature_params"]

            col = "preprocessing_params" if "preprocessing_params" in info_json else "preparation_params"
            feature_class = str_to_class[info_json[col]["class"]]
            feature_nature = info_json[col]["nature"]
            feature_label = info_json[col]["name"]

            rir_tr = info_json["preparation_params"][
                         "rir_likelihood"] > 0.0 if "preparation_params" in info_json else False
            vol_tr = (True if info_json["preprocessing_params"]["cdf_volume_target"] else False) if (
                    "preprocessing_params" in info_json) else False

            if feature_class == ClassFlags.NOTHOTWORD:
                feature_class = explode_not_hot_word_class(feature_label)

            # is training data
            if feat_params["is_training"]:

                # features of this nature are excluded entirely
                if feature_nature not in listing.keys():
                    continue

                classes_and_filters = listing[feature_nature]

                for feat_desc in classes_and_filters:
                    # mismatching feature class
                    if feature_class not in feat_desc.classes:
                        continue

                    # label filter present and not matching label
                    if feat_desc.label_filters and not any(lf in feature_label for lf in feat_desc.label_filters):
                        continue

                    # check if rir and vol transform flags are matching
                    if feat_desc.rir_tr != rir_tr or feat_desc.vol_tr != vol_tr:
                        continue

                    if feature_class == ClassFlags.HOT_WORD:
                        train_feat.append((root, 1))

                        hotword_sample_composition[nature_idx[feature_nature]] += feat_params["n_samples"]
                    else:
                        train_feat.append((root, 0))

                    if feat_desc.feature_filters:
                        train_filter.append(feat_desc.feature_filters)
                    else:
                        train_filter.append((-1, -1, -1, -1))

            # not training data -> test data (without filtering)
            else:
                test_feat.append((root, 1 if feature_class == ClassFlags.HOT_WORD else 0))

            # ensure consistency across features
            expected_feat_win_length = expected_feat_win_length or feat_params["extraction"]["feat_win_len"]
            expected_feat_hop_length = expected_feat_hop_length or feat_params["extraction"]["feat_hop_len"]
            expected_feat_type = expected_feat_type or feat_params["feature"]
            expected_feat_size = expected_feat_size or feat_params["size"]

            assert expected_feat_win_length == feat_params["extraction"]["feat_win_len"]
            assert expected_feat_hop_length == feat_params["extraction"]["feat_hop_len"]
            assert expected_feat_type == feat_params["feature"]
            assert expected_feat_size == feat_params["size"]

    hotword_sample_total = sum(hotword_sample_composition)
    train_data_stub = ""

    # add nature of hotword samples using in training
    for key in nature_idx:
        if hotword_sample_composition[nature_idx[key]] > 0:
            perc = hotword_sample_composition[nature_idx[key]] / hotword_sample_total
            train_data_stub += f"{round(perc * 100)}{key}"

    # add feature configuration
    if expected_feat_size and expected_feat_type and expected_feat_win_length and expected_feat_hop_length:
        train_data_stub += f"_{expected_feat_size[2]}_{expected_feat_size[1]}_{expected_feat_type.upper()}"
        # TODO  assuming 16kHz, as this field is not present in legacy feature dirs. Maybe change later
        train_data_stub += f"_W{expected_feat_win_length // 16}_H{expected_feat_hop_length // 16}"

    return train_feat, train_filter, test_feat, train_data_stub


def play_audio_file(path: str) -> None:
    wave, sr = torchaudio.load(path)
    audio = pydub.AudioSegment(wave.numpy().tobytes(),
                               frame_rate=sr,
                               sample_width=4,
                               channels=1)

    pydub.playback.play(audio)


def play_dir(source_dir: str) -> None:
    files = get_audio_files_by_dir(source_dir)
    for file in files:
        play_audio_file(file)


def convert_to_wave(path: str) -> None:
    wave, sr = torchaudio.load(path)
    wave, _ = normalize(wave)
    torchaudio.save(f"{path[:path.rfind('.')]}.wav", wave, sr)
