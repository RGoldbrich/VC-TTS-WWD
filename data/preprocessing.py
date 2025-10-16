# standard lib
import csv
import json
import logging
import math
import multiprocessing as mp
import os
import random
import shutil
from time import time

# third party
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader

# application
import common.audio as audioc
import common.log as log
from common.cdf_helper import BetterVolCdfUtil
from insight import find_speech_bounds


def prepare_audio(
        wave: torch.Tensor,
        sr_source: int,
        sr_target: int,
        device: str = "cpu"
) -> torch.Tensor:
    """
    Convert wave to mono (if not already the case), drops the channel dimension and resamples to target sampling rate.
    Moves wave tensor to device.
    :param wave: input wave
    :param sr_source: original sampling rate of input wave
    :param sr_target: target sampling rate
    :param device:
    :return:
    """
    # convert to mono when necessary and drop channel dimension
    wave = audioc.to_mono_drop_ch_dim(wave)

    # resample to target frequency
    if sr_source != sr_target:
        return torchaudio.functional.resample(wave.to(device), sr_source, sr_target)
    else:
        return wave.to(device)


def contaminate(
        wave: torch.Tensor,
        sr: int,
        cont_sets: list[tuple[list[str], float, float, float]],
        device: str = "cpu",
) -> tuple[torch.Tensor, tuple] | tuple[torch.Tensor, bool]:
    """
    Augments audio by overlaying additional audio
    :param wave: input wave
    :param sr: sampling rate of input wave
    :param cont_sets: contamination settings:
    list of file-list, contamination likelihood, snr lower bound, snr upper bound
    :param device:
    :return:
    """
    random_value = random.random()
    for idx, cont_set in enumerate(cont_sets):
        # subtract likelihood from random value
        random_value -= cont_set[1]

        # if random value falls below zero, contaminated using this dataset
        if random_value < 0:
            logging.info(f"contaminating using dataset {idx} (change of {cont_set[1]:.2%})")

            # load random file
            full_file = random.choice(cont_set[0])
            logging.info(f"random file for contamination: {full_file}")
            wave_cont, sr_cont = torchaudio.load(full_file)

            # convert to mono, drop channel dim and resample
            wave_cont = prepare_audio(wave_cont, sr_cont, sr, device)

            # cut/extent to match number of frames with random center
            wave_cont, front_cut = audioc.cut_extend_wave_randomly(wave_cont, len(wave))

            # overlay using specified signal-to-noise ration
            snr = random.uniform(cont_set[2], cont_set[3])

            # add channel dimension (required by add_noise())
            wave = wave[None]
            wave_cont = wave_cont[None]

            wave = torchaudio.functional.add_noise(
                wave.to(device),
                wave_cont.to(device),
                torch.Tensor([snr]).to(device))

            # wave = wave.to("cpu")
            wave = wave[0]  # remove channel dimension again

            cont_info = (
                # file used for contamination
                full_file,
                # begin of audio sections used for contamination (can be negative if audio needed padding)
                front_cut,
                # signal-to-noise ration used
                snr)

            return wave, cont_info

    logging.info(f"Not contaminating audio; returning original")

    return wave, False


def apply_rir(
        wave: torch.Tensor,
        sr: int,
        rir_files: list[str],
        device: str = "cpu",
) -> torch.Tensor:
    """
    Applies room impulse response provided by sample audio in rir_files
    :param wave: input wave
    :param sr: sampling rate of input wave
    :param rir_files: list of full file paths to rir example audio files
    :param device:
    :return: wave
    """
    # choose random rir file
    rir_file = random.choice(rir_files)
    logging.info(f"applying room impulse response using {rir_file}")

    # load and prepare
    wave_rir, sr_rir = torchaudio.load(rir_file)
    wave_rir = prepare_audio(wave_rir, sr_rir, sr, device)

    # apply rir and cut again
    wave = torchaudio.functional.convolve(wave.to(device), wave_rir.to(device))[:len(wave)]

    return wave


def preprocess_audios(
        files: list[str],
        global_start_idx: int,
        out_dir: str,
        sr_target: int,
        n_frames: int,
        hop_length_ms: int,
        ensure_single: str,
        expl_contamination_settings: list[tuple[list[str], float, float, float]],
        cdf_volume_transform: tuple[str, str],
        result_list,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cdf_volume_transform is not None:
        # use db_max instead of content_db_max here
        vol_transform = BetterVolCdfUtil(cdf_volume_transform[0], cdf_volume_transform[1], "vol_max_db")
    else:
        vol_transform = None

    ts_begin = time()

    info = []
    filtered = 0

    for idx, file in enumerate(files):
        # load audio
        try:
            wave, sr = torchaudio.load(file)
        except Exception:
            log.print_log(f"{mp.current_process().name}: cannot open {file}")
            continue

        # compute max db here and not later, resampling slightly changes volume
        wave_max_db = audioc.max_db(wave)

        # convert to mono, drop channel dim and resample
        wave = prepare_audio(wave, sr, sr_target, device)
        if len(wave) == 0:
            # logging.warning("length of wave is 0")
            continue

        # determine hop length in frames and number of chunks that can be extracted
        hop_length_frames = int(sr_target * hop_length_ms / 1000)
        n_chunks = math.ceil((len(wave) - n_frames) / hop_length_frames)

        # soft filtering using vad: if audio is too long -> try and shorten it,
        # update n_chunks and test again: if still too long -> discard
        if n_chunks > 0 and ensure_single == "Soft":
            begin, end = find_speech_bounds(wave.to("cpu"), sr)
            wave = wave[begin:end]
            n_chunks = math.ceil((len(wave) - n_frames) / hop_length_frames)

            if n_chunks > 0:
                filtered += 1
                continue

            wave = wave.to(device)

        # hard filtering using audio file length: if too long -> discard
        if n_chunks > 0 and ensure_single == "Hard":
            filtered += 1
            continue

        if n_chunks < 1:
            # logging.info("audio shorter than target length; creating single, padded audio")
            wave, _ = audioc.cut_extend_wave_randomly(wave, n_frames)

            # match to target volume cdf
            if vol_transform:
                # ensure no resample was needed (which would have messed with the audios volume)
                assert sr == sr_target

                try:
                    target_vol = vol_transform.convert_vol(file.split('/')[-1])
                except Exception:
                    continue

                wave = torchaudio.functional.gain(wave, target_vol - wave_max_db)

            # contaminate
            if expl_contamination_settings:
                wave, cont_info = contaminate(wave, sr_target, expl_contamination_settings, device)
            else:
                cont_info = False

            out_file = f"{idx + global_start_idx:05d}_padded.wav"
            torchaudio.save(f"{out_dir}/{out_file}", wave.to('cpu')[None], sr_target)

            info.append((
                file.split('/')[-1],  # filename
                cont_info,  # info about contamination/augmentation
                out_file,  # out filename
            ))
        else:
            # match to target volume cdf
            if vol_transform:
                # ensure no resample was needed (which would have messed with the audios volume)
                assert sr == sr_target

                try:
                    target_vol = vol_transform.convert_vol(file.split('/')[-1])
                except Exception:
                    continue

                wave = torchaudio.functional.gain(wave, target_vol - wave_max_db)

            # contaminate once for the entire length of the file
            # this does lower diversity, but saves time
            if expl_contamination_settings:
                wave, cont_info = contaminate(wave, sr_target, expl_contamination_settings, device)
            else:
                cont_info = False

            # randomly use remaining frames
            # do not align first chunk with beginning of audio and leave unused audio towards the back
            extra_space = int(len(wave) - n_frames - (hop_length_frames * (n_chunks - 1)))
            offset = random.randrange(extra_space) if extra_space else 0

            for c in range(n_chunks):
                wave_chunk = wave[c * hop_length_frames + offset: c * hop_length_frames + n_frames + offset]
                assert len(wave_chunk) == n_frames

                out_file = f"{idx + global_start_idx:05d}_{c:04d}.wav"
                torchaudio.save(f"{out_dir}/{out_file}", wave_chunk.to('cpu')[None], sr_target)

                info.append((
                    file.split('/')[-1],  # filename
                    cont_info,  # info about contamination/augmentation
                    out_file,  # out filename
                ))

        log.log_time(ts_begin, idx, len(files), 50, True)

    print("filtered out:", filtered)
    result_list.extend(info)


def prepare_audios(
        files: list[str],
        global_start_idx: int,
        out_dir: str,
        sr_target: int,
        expl_contamination_settings: list[tuple[list[str], float, float, float]],
        rir_files: list[str],
        rir_likelihood: float,
        result_list,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ts_begin = time()

    info = []

    for idx, file in enumerate(files):
        # load audio
        try:
            wave, sr = torchaudio.load(file)
        except Exception:
            log.print_log(f"{mp.current_process().name}: cannot open {file}")
            continue

        # convert to mono, drop channel dim and resample
        wave = prepare_audio(wave, sr, sr_target, device)
        if len(wave) == 0:
            # logging.warning("length of wave is 0")
            continue

        # apply rir
        if random.random() < rir_likelihood:
            wave = apply_rir(wave, sr_target, rir_files, device)

            # if amplitudes are > 1.0 after rir, normalize audio to avoid clipping on torchaudio.save()
            if torch.max(wave) > 1.0:
                wave, _ = audioc.normalize(wave)

        # contaminate
        if expl_contamination_settings:
            wave, cont_info = contaminate(wave, sr_target, expl_contamination_settings, device)
        else:
            cont_info = False

        out_file = f"{idx + global_start_idx:05d}.wav"
        torchaudio.save(f"{out_dir}/{out_file}", wave.to('cpu')[None], sr_target)

        info.append((
            file.split('/')[-1],  # filename
            cont_info,  # info about contamination/augmentation
            out_file,  # out filename
        ))

        log.log_time(ts_begin, idx, len(files), 50, True)

    result_list.extend(info)


def run(
        source_dir: str,
        labels: (str, str, str),
        mode: str = "preprocess",
        sr_target: int = 16000,
        length_ms: int = 2400,
        hop_length_ms: int = 1200,
        ensure_single: str = "Not",
        contamination_settings: list[tuple[str, float, float, float]] = None,
        cdf_volume_target: str = None,
        rir_likelihood: float = 0.0,
        n_processes: int = 4
) -> None:
    """
    Main entry function for all wave level preprocessing. This inputs and outputs .wav files
    :param source_dir: directory to recursively gather audio files from
    :param labels:
        tuple containing (class, nature, name), with class in
        ["HotWord", "NotHotWord", "Speech", "Ambient", "Raw"] and nature in ["TTS", "VC", "HUM", "AMB"]
    :param mode: defines what steps are used in pipeline, must be in ["prepare", "preprocess"]
    :param sr_target: target sampling rate
    :param length_ms: length to cut to (only for mode "preprocess")
    :param hop_length_ms:
        hop length for extracting overlapping sections from source audio (only for mode in ["preprocess"])
    :param ensure_single:
        whether and how to ensure a single file is produced,
        with ensure_single in ["Hard", "Soft", "Not"]. "Hard" uses file length and filters files longer than the
        specified number of frames; "Soft" uses Vad to try find short enough speech section; "Not" uses hop_length_ms to
        extract overlapping sections from source.
    :param contamination_settings: defines augmentation dataset to use (directory, likelihood, lower-snr, upper-snr)
    :param cdf_volume_target:
        used to match volume to target cdf, assumes info.csv with volume info present in source
        (only for mode "preprocess")
    :param rir_likelihood: likelihood of applying room impulse response (only for mode "prepare")
    :param n_processes: number of processes to use
    """
    if contamination_settings is None:
        contamination_settings = []

    # load or create info json and store function parameters
    locals_temp = locals().copy()  # needed to avoid circular reference
    parameter_json = log.get_or_create_parameter_json(source_dir)

    # check for provided class, nature and stub for validity
    class_label, nature_label, name_label = labels
    if class_label not in ["HotWord", "NotHotWord", "Speech", "Ambient", "Raw"]:
        raise Exception(f"Class label {class_label} is not supported")

    if nature_label not in ["TTS", "VC", "HUM", "AMB"]:
        raise Exception(f"Nature label {nature_label} is not supported")

    if '_' in name_label:
        raise Exception(f"'_' not allowed in name label")
    if name_label[0].islower():
        raise Exception(f"name label should start with capital letter")

    # create output directory and log parameters
    out_dir = log.setup_out_dir(os.getcwd(), "preparation", "_".join(labels))
    log_handler = log.setup_logging(out_dir + "/func.log")
    logging.info("start of prepare_audios")
    logging.info(f"prepare_audios() params: {locals()}")

    # determine mode
    if mode == "preprocess":
        logging.info("running in preprocess data mode")
    elif mode == "prepare":
        logging.info("running in prepare mode")
    else:
        msg = "cannot determine mode"
        logging.error(msg)
        raise Exception(msg)

    # gather all files from this directory and all its subdirectory
    files = [os.path.join(root, file) for root, _, files in os.walk(source_dir) for file in files if
             any(file.lower().endswith(ext) for ext in audioc.EXTENSIONS)]
    n_files = len(files)
    logging.info(f"{n_files} files found in directory")

    # load filenames used for contamination and keep contamination likelihood and signal-to-noise ratio range
    logging.info(f"contaminating using {len(contamination_settings)} dataset(s)")
    expl_contamination_settings = []
    for idx, cont_set in enumerate(contamination_settings):
        # create list of all files in contamination dataset
        cont_files = [os.path.join(root, file) for root, _, files in os.walk(cont_set[0]) for file in files if
                      any(file.lower().endswith(ext) for ext in audioc.EXTENSIONS)]

        logging.info(f"{idx}: {cont_set[0]} with likelihood of {cont_set[1]:.2%} "
                     f"and signal to noise ratio range of {cont_set[2]:+} to {cont_set[3]:+}")
        expl_contamination_settings.append((cont_files, cont_set[1], cont_set[2], cont_set[3]))

    # number of frames to cut/extent the clips to in order to achieve desired length at specified sampling rate
    n_frames = int(length_ms * sr_target / 1000)
    logging.info(f"cutting to {n_frames} frames at {sr_target} Hz for {length_ms / 1000:.2f}s clip length")

    files_per_proc = n_files // n_processes

    # load filename used for room impulse response
    rir_files = [os.path.join(root, file) for root, _, files in os.walk(audioc.DS_DIR + audioc.RIR)
                 for file in files
                 if any(file.lower().endswith(ext) for ext in audioc.EXTENSIONS)]

    # preprocessing mode:
    # resamples to target (sr) if not already the case,
    # cuts/pads to desired length or extracts overlapping sections if audio is longer then desired length,
    # optionally filters out audio longer then the number of frames specified
    # applies volume transform,
    # augments audio with environmental audio,
    if mode == "preprocess":
        parameter_json["preprocessing_params"] = locals_temp
        parameter_json["preprocessing_params"]["class"] = class_label
        parameter_json["preprocessing_params"]["nature"] = nature_label
        parameter_json["preprocessing_params"]["name"] = name_label
        parameter_json["preprocessing_params"]["n_files"] = n_files

        cdf_volume_transform = (os.path.join(source_dir, "info.csv"), cdf_volume_target) if cdf_volume_target else None

        with mp.Manager() as manager:
            results_list = manager.list()

            processes = []
            for pid in range(n_processes):
                start_idx = pid * files_per_proc
                end_idx = (pid + 1) * files_per_proc if pid != n_processes - 1 else n_files
                p = mp.Process(target=preprocess_audios,
                               args=(files[start_idx:end_idx],
                                     start_idx,
                                     out_dir,
                                     sr_target,
                                     n_frames,
                                     hop_length_ms,
                                     ensure_single,
                                     expl_contamination_settings,
                                     cdf_volume_transform,
                                     results_list))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            info = list(results_list)

    # preparation mode:
    # resamples to target sr, and applies rir if specified and augments audio with environmental audio
    # DOES NOT alter the length of audio clips
    elif mode == "prepare":
        parameter_json["preparation_params"] = locals_temp
        parameter_json["preparation_params"]["class"] = class_label
        parameter_json["preparation_params"]["nature"] = nature_label
        parameter_json["preparation_params"]["name"] = name_label
        parameter_json["preparation_params"]["n_files"] = n_files

        with mp.Manager() as manager:
            results_list = manager.list()

            processes = []
            for pid in range(n_processes):
                start_idx = pid * files_per_proc
                end_idx = (pid + 1) * files_per_proc if pid != n_processes - 1 else n_files
                p = mp.Process(target=prepare_audios,
                               args=(files[start_idx:end_idx],
                                     start_idx,
                                     out_dir,
                                     sr_target,
                                     expl_contamination_settings,
                                     rir_files,
                                     rir_likelihood,
                                     results_list))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            info = list(results_list)

    # append to existing info.csv (present if this dataset was generated using TTS or VC methods)
    if os.path.exists(os.path.join(source_dir, "info.csv")):
        logging.info(f"appending to existing info.csv")
        existing_info_csv = pd.read_csv(os.path.join(source_dir, "info.csv"))
        info_df = pd.DataFrame(info, columns=["filename", f"{mode}_cont_info", f"{mode}_filename"])

        if mode == "prepare":
            combined_df = pd.merge(existing_info_csv, info_df, on="filename")
        elif mode == "preprocess":
            # if this is preprocessing, the loaded file
            # (filename in info and info_df) is the same as prepare_filename in csv
            combined_df = pd.merge(
                existing_info_csv, info_df,
                left_on="prepare_filename", right_on="filename", suffixes=("", "_y")  # both dfs contain "filename"
            )
            combined_df = combined_df.drop(columns=["filename_y"])
        else:
            raise Exception()

        combined_df.to_csv(f"{out_dir}/info.csv", index=False)

    # write to new info.csv
    else:
        logging.info(f"writing new info.csv")
        with open(f"{out_dir}/info.csv", mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(("filename", f"{mode}_cont_info", f"{mode}_filename"))
            writer.writerows(info)

    # copy alignment json if present
    if os.path.exists(os.path.join(source_dir, "alignment.json")):
        logging.info(f"copying alignment.json from {source_dir} to {out_dir}")
        shutil.copyfile(os.path.join(source_dir, "alignment.json"), os.path.join(out_dir, "alignment.json"))

    # copy confidence json if present
    if os.path.exists(os.path.join(source_dir, "confidence.json")):
        logging.info(f"copying confidence.json from {source_dir} to {out_dir}")
        shutil.copyfile(os.path.join(source_dir, "confidence.json"), os.path.join(out_dir, "confidence.json"))

    # write info json about this preprocessing folder
    log.store_parameter_json(out_dir, parameter_json)

    logging.info("end of prepare_audios")

    log.close_logging(log_handler)


def prepare_main():
    d = "cuda" if torch.cuda.is_available() else "cpu"
    if d == "cuda":
        mp.set_start_method("spawn")

    # synthetic datasets
    run("../Datasets/synthetic_data/TTS_Raw_VCTK", ("Raw", "TTS", "VCTK"), "prepare", rir_likelihood=0.0)
    run("../Datasets/synthetic_data/TTS_Raw_VCTK", ("Raw", "TTS", "VCTKRir"), "prepare", rir_likelihood=1.0)

    # run("../Datasets/synthetic_data/TTS_Raw_Libri", ("Raw", "TTS", "LibriRir"), "prepare", rir_likelihood=1.0)
    # run("../Datasets/synthetic_data/VC_Raw_PeopleSpeech", ("Raw", "VC", "PeopleSpeechRir"), "prepare", rir_likelihood=1.0)
    # run("../Datasets/synthetic_data/VC_Raw_Mcv", ("Raw", "VC", "McvRir"), "prepare", rir_likelihood=1.0)
    # run("../Datasets/synthetic_data/VC_Raw_TedLium", ("Raw", "VC", "TedLiumRir"), "prepare", rir_likelihood=1.0)

    # real underlying source datasets
    run(audioc.VCTK_S, ("Speech", "HUM", "VCTK"), "prepare", rir_likelihood=0.0)
    run(audioc.VCTK_S, ("Speech", "HUM", "VCTKRir"), "prepare", rir_likelihood=1.0)
    # run(audioc.LIBRI_Speech_S, ("Speech", "HUM", "LibriRir"), "prepare", rir_likelihood=1.0)
    # run(audioc.PEOPLE_Speech_S, ("Speech", "HUM", "PeopleSpeechRir"), "prepare", rir_likelihood=1.0)
    # run(audioc.MCV_Speech_S, ("Speech", "HUM", "McvRir"), "prepare", rir_likelihood=1.0)
    # run(audioc.TEDLIUM_S, ("Speech", "HUM", "TedLiumRir"), "prepare", rir_likelihood=1.0)

    # sonos
    # run("../Datasets/synthetic_data/VC_Raw_Sonos", ("Raw", "VC", "Sonos"), "prepare")
    # run("../Datasets/synthetic_data/VC_Raw_Sonos", ("Raw", "VC", "SonosRir"), "prepare", rir_likelihood=1.0)


def realref_main():
    d = "cuda" if torch.cuda.is_available() else "cpu"
    if d == "cuda":
        mp.set_start_method("spawn")

    cont_setts = [(audioc.MUSAN_Train, 1.0, 5, 5)]

    # run(audioc.SONOS_SNIPS_HotWord_Train, ("HotWord", "HUM", "SonosTrain"), "prepare", rir_likelihood=0.0)

    run("out_preparation/250201_193214_HotWord_HUM_SonosHeySnips", ("HotWord", "HUM", "SonosTrain"), "preprocess",
        contamination_settings=cont_setts, ensure_single="Hard")
    # run(audioc.SONOS_SNIPS_Speech_Train, ("Speech", "HUM", "SonosTrain"), "preprocess",
    #     contamination_settings=cont_setts, hop_length_ms=250)
    #
    cont_setts = [(audioc.MUSAN_Test, 1.0, 5, 5)]
    cont_setts = None

    # run(audioc.SONOS_SNIPS_HotWord_Test, ("HotWord", "HUM", "SonosCleanTest"), "prepare",
    #     contamination_settings=cont_setts)
    # run(audioc.SONOS_SNIPS_Speech_Test, ("Speech", "HUM", "SonosCleanTest"), "prepare",
    #     contamination_settings=cont_setts)


def aux_preprocess_main():
    d = "cuda" if torch.cuda.is_available() else "cpu"
    if d == "cuda":
        mp.set_start_method("spawn")

    cont_setts = [(audioc.MUSAN_Train, 1.0, 5, 5)]

    cdf_volume_target = "../Projects/SSD-KWS/data/out_inspect/241128_135512_SonosTrainSpeech/info.csv"

    run("../Projects/SSD-KWS/data/out_preparation/a1_prepared/250201_114704_Speech_HUM_VCTK",
        ("Speech", "HUM", "VCTK"), "preprocess", contamination_settings=cont_setts,
        hop_length_ms=250,
        cdf_volume_target=cdf_volume_target)

    run("../Projects/SSD-KWS/data/out_preparation/a1_prepared/250201_114704_Speech_HUM_VCTK",
        ("Speech", "HUM", "VCTKNv"), "preprocess", contamination_settings=cont_setts,
        hop_length_ms=250,
        cdf_volume_target=None)

    run("../Projects/SSD-KWS/data/out_preparation/a1_prepared_rir/250201_114718_Speech_HUM_VCTKRir",
        ("Speech", "HUM", "VCTKRir"), "preprocess", contamination_settings=cont_setts,
        hop_length_ms=250,
        cdf_volume_target=cdf_volume_target)

    run("../Projects/SSD-KWS/data/out_preparation/a1_prepared_rir/250201_114718_Speech_HUM_VCTKRir",
        ("Speech", "HUM", "VCTKRirNv"), "preprocess", contamination_settings=cont_setts,
        hop_length_ms=250,
        cdf_volume_target=None)

    # run("../Projects/SSD-KWS/data/out_preparation/a1_prepared/241203_113319_Speech_HUM_Libri",
    #     ("Speech", "HUM", "LibriNv"), "preprocess", contamination_settings=cont_setts,
    #     hop_length_ms=450,
    #     cdf_volume_target=None)
    #
    # run("../Projects/SSD-KWS/data/out_preparation/a1_prepared/241128_120622_Speech_HUM_PeopleSpeech",
    #     ("Speech", "HUM", "PeopleSpeechNv"), "preprocess", contamination_settings=cont_setts,
    #     hop_length_ms=250,
    #     cdf_volume_target=None)
    #
    # run("../Projects/SSD-KWS/data/out_preparation/a1_prepared/241201_113722_Speech_HUM_Mcv",
    #     ("Speech", "HUM", "McvNv"), "preprocess", contamination_settings=cont_setts,
    #     hop_length_ms=250,
    #     cdf_volume_target=None)
    #
    # run("../Projects/SSD-KWS/data/out_preparation/a1_prepared/241201_192118_Speech_HUM_TedLium",
    #     ("Speech", "HUM", "TedLiumNv"), "preprocess", contamination_settings=cont_setts,
    #     hop_length_ms=450,
    #     cdf_volume_target=None)
    #
    # run("../Projects/SSD-KWS/data/out_preparation/a1_prepared_rir/241213_190417_Speech_HUM_LibriRir",
    #     ("Speech", "HUM", "LibriRirNv"), "preprocess", contamination_settings=cont_setts,
    #     hop_length_ms=450,
    #     cdf_volume_target=None)
    #
    # run("../Projects/SSD-KWS/data/out_preparation/a1_prepared_rir/241213_192445_Speech_HUM_PeopleSpeechRir",
    #     ("Speech", "HUM", "PeopleSpeechRirNv"), "preprocess", contamination_settings=cont_setts,
    #     hop_length_ms=250,
    #     cdf_volume_target=None)
    #
    # run("../Projects/SSD-KWS/data/out_preparation/a1_prepared_rir/241213_193817_Speech_HUM_McvRir",
    #     ("Speech", "HUM", "McvRirNv"), "preprocess", contamination_settings=cont_setts,
    #     hop_length_ms=250,
    #     cdf_volume_target=None)
    #
    # run("../Projects/SSD-KWS/data/out_preparation/a1_prepared_rir/241214_122041_Speech_HUM_TedLiumRir",
    #     ("Speech", "HUM", "TedLiumRirNv"), "preprocess", contamination_settings=cont_setts,
    #     hop_length_ms=450,
    #     cdf_volume_target=None)


def synth_preprocess_main():
    d = "cuda" if torch.cuda.is_available() else "cpu"
    if d == "cuda":
        mp.set_start_method("spawn")

    cont_setts = [(audioc.MUSAN_Train, 1.0, 5, 5)]

    cdf_volume_target = "../Projects/SSD-KWS/data/out_inspect/241128_135434_SonosTrainHotWord/info.csv"

    # TTS VCTK
    run("../Projects/SSD-KWS/data/out_preparation/s2_synthetic_split_rir/250216_152345_NotHotWord_TTS_VCTKRirHey",
        ("NotHotWord", "TTS", "VCTKRirHey"), "preprocess", ensure_single="Hard",
        contamination_settings=cont_setts,
        cdf_volume_target=cdf_volume_target)
    run("../Projects/SSD-KWS/data/out_preparation/s2_synthetic_split_rir/250216_152315_NotHotWord_TTS_VCTKRirSnips",
        ("NotHotWord", "TTS", "VCTKRirSnips"), "preprocess", ensure_single="Hard",
        contamination_settings=cont_setts,
        cdf_volume_target=cdf_volume_target)

    # TTS Libri
    run("../Projects/SSD-KWS/data/out_preparation/s2_synthetic_split_rir/250216_152439_NotHotWord_TTS_LibriRirHey",
        ("NotHotWord", "TTS", "LibriRirHey"), "preprocess", ensure_single="Hard",
        contamination_settings=cont_setts,
        cdf_volume_target=cdf_volume_target)
    run("../Projects/SSD-KWS/data/out_preparation/s2_synthetic_split_rir/250216_152410_NotHotWord_TTS_LibriRirSnips",
        ("NotHotWord", "TTS", "LibriRirSnips"), "preprocess", ensure_single="Hard",
        contamination_settings=cont_setts,
        cdf_volume_target=cdf_volume_target)

    # VC Mcv
    run("../Projects/SSD-KWS/data/out_preparation/s2_synthetic_split_rir/250216_152058_NotHotWord_VC_McvRirHey",
        ("NotHotWord", "VC", "McvRirHey"), "preprocess", ensure_single="Hard",
        contamination_settings=cont_setts,
        cdf_volume_target=cdf_volume_target)
    run("../Projects/SSD-KWS/data/out_preparation/s2_synthetic_split_rir/250216_152026_NotHotWord_VC_McvRirSnips",
        ("NotHotWord", "VC", "McvRirSnips"), "preprocess", ensure_single="Hard",
        contamination_settings=cont_setts,
        cdf_volume_target=cdf_volume_target)

    # VC PeopleSpeech
    run("../Projects/SSD-KWS/data/out_preparation/s2_synthetic_split_rir/250216_152155_NotHotWord_VC_PeopleSpeechRirHey",
        ("NotHotWord", "VC", "PeopleSpeechRirHey"), "preprocess", ensure_single="Hard",
        contamination_settings=cont_setts,
        cdf_volume_target=cdf_volume_target)
    run("../Projects/SSD-KWS/data/out_preparation/s2_synthetic_split_rir/250216_152125_NotHotWord_VC_PeopleSpeechRirSnips",
        ("NotHotWord", "VC", "PeopleSpeechRirSnips"), "preprocess", ensure_single="Hard",
        contamination_settings=cont_setts,
        cdf_volume_target=cdf_volume_target)

    # VC TedLium
    run("../Projects/SSD-KWS/data/out_preparation/s2_synthetic_split_rir/250216_152251_NotHotWord_VC_TedLiumRirHey",
        ("NotHotWord", "VC", "TedLiumRirHey"), "preprocess", ensure_single="Hard",
        contamination_settings=cont_setts,
        cdf_volume_target=cdf_volume_target)
    run("../Projects/SSD-KWS/data/out_preparation/s2_synthetic_split_rir/250216_152221_NotHotWord_VC_TedLiumRirSnips",
        ("NotHotWord", "VC", "TedLiumRirSnips"), "preprocess", ensure_single="Hard",
        contamination_settings=cont_setts,
        cdf_volume_target=cdf_volume_target)
