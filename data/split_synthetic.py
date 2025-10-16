# standard lib
import json
import logging
import os
import shutil
from time import time

# third party
import pandas as pd
import torch
import torchaudio

# application
from alignment import clean_transcript
from common import log as log


def split_synthetic(source_dir: str, sub_phrase: str, labels: (str, str, str), padding_s: int = 0.05):
    locals_temp = locals().copy()  # needed to avoid circular reference
    parameter_json = log.get_or_create_parameter_json(source_dir)
    parameter_json["split"] = locals_temp

    if sub_phrase not in ["hey snips", "hey", "snips", "hey _", "_ snips", "front", "back"]:
        raise Exception(f"sub_phrase {sub_phrase} is not supported")

    # check for provided class, nature and stub for validity
    class_label, nature_label, name_label = labels
    if class_label not in ["HotWord", "NotHotWord", "Speech"]:
        raise Exception(f"Class label {class_label} is not supported")

    if nature_label not in ["TTS", "VC", "HUM"]:
        raise Exception(f"Nature label {nature_label} is not supported")

    if '_' in name_label:
        raise Exception(f"'_' not allowed in name label")
    if name_label[0].islower():
        raise Exception(f"name label should start with capital letter")

    # output dir and logging
    out_dir = log.setup_out_dir(os.getcwd(), "preparation", "_".join(labels))
    log_handler = log.setup_logging(out_dir + "/split.log")
    logging.info("start of split_synthetic")
    logging.info(f"split_synthetic() params: {locals()}")

    # load info csv
    df = pd.read_csv(os.path.join(source_dir, "info.csv"))
    # load word-level alignment info
    with open(os.path.join(source_dir, "alignment.json"), "r") as f:
        alignments = json.load(f)
    # load confidence info
    with open(os.path.join(source_dir, "confidence.json"), "r") as f:
        confidences = json.load(f)
    phon_key = "phoneme_gen_mean_p2"  # using generalized mean with p=2
    fill_key = "filler_gen_mean_p-2"  # using generalized mean with p=-2 (stronger than harmonic mean)

    info = []

    ts_begin = time()

    for index, row in df.iterrows():
        wave, sr = torchaudio.load(os.path.join(source_dir, row["prepare_filename"]))
        alignment = alignments[row["filename"]]
        confidence = confidences[row["filename"]]

        transcript = clean_transcript(row["transcript"]).split()
        assert len(alignment) == len(transcript)

        hey_index = transcript.index("hey")
        snips_index = transcript.index("snips")
        assert hey_index + 1 == snips_index

        duration = 0

        if sub_phrase == "hey snips":
            start_frame = max(int((alignment[hey_index][0] - padding_s) * sr), 0)  # start of hey
            end_frame = min(int((alignment[hey_index + 1][1] + padding_s) * sr), wave.shape[-1])  # end of snips

            wave_section = wave[:, start_frame: end_frame]

            duration = (end_frame - start_frame) / sr

            phrase_confidence = confidence["hey snips"][phon_key] * confidence["hey snips"][fill_key]

        elif sub_phrase == "hey":
            start_frame = int((alignment[hey_index][0] - padding_s) * sr)  # start of hey
            end_frame = int((alignment[hey_index][1]) * sr)  # end of hey

            wave_section = wave[:, start_frame: end_frame]

            duration = (end_frame - start_frame) / sr

            phrase_confidence = confidence["hey"][phon_key] * confidence["hey"][fill_key]

        elif sub_phrase == "snips":
            start_frame = int((alignment[hey_index + 1][0]) * sr)  # start of snips
            end_frame = int((alignment[hey_index + 1][1] + padding_s) * sr)  # end of snips

            wave_section = wave[:, start_frame: end_frame]

            duration = (end_frame - start_frame) / sr

            phrase_confidence = confidence["snips"][phon_key] * confidence["snips"][fill_key]

        elif sub_phrase == "hey _":
            start_frame = int((alignment[hey_index][0] - padding_s) * sr)  # start of hey
            end_frame = int((alignment[hey_index][1] + padding_s) * sr)  # end of hey

            wave_hey = wave[:, start_frame: end_frame]

            duration = (end_frame - start_frame) / sr

            start_frame = int((alignment[hey_index + 2][0] - padding_s) * sr)  # start of first word after snips
            end_frame = int((alignment[hey_index + 3][1] + padding_s) * sr)  # end of second word after snips

            wave_following_words = wave[:, start_frame: end_frame]

            wave_section = torch.concat([wave_hey, wave_following_words], dim=1)

            # using confidence in placing the entire "hey snips"
            phrase_confidence = confidence["hey snips"][phon_key] * confidence["hey snips"][fill_key]

        elif sub_phrase == "_ snips":
            start_frame = int((alignment[hey_index + 1][0] - padding_s) * sr)  # start of snips
            end_frame = int((alignment[hey_index + 1][1] + padding_s) * sr)  # end of snips

            wave_snips = wave[:, start_frame: end_frame]

            duration = (end_frame - start_frame) / sr

            start_frame = int((alignment[hey_index - 2][0] - padding_s) * sr)  # start of second word before hey
            end_frame = int((alignment[hey_index - 1][1] + padding_s) * sr)  # end of first word before hey

            wave_preceding_words = wave[:, start_frame: end_frame]

            wave_section = torch.concat([wave_preceding_words, wave_snips], dim=1)

            # using confidence in placing the entire "hey snips"
            phrase_confidence = confidence["hey snips"][phon_key] * confidence["hey snips"][fill_key]

        elif sub_phrase == "front":
            end_frame = int((alignment[hey_index - 1][1] + padding_s) * sr)  # end of first word before hey

            wave_section = wave[:, :end_frame]

            # using confidence in placing the "hey"
            phrase_confidence = confidence["hey"][phon_key] * confidence["hey"][fill_key]

        elif sub_phrase == "back":
            start_frame = int((alignment[hey_index + 2][0] - padding_s) * sr)  # start of fist word after snips

            wave_section = wave[:, start_frame:]

            # using confidence in placing the "snips"
            phrase_confidence = confidence["snips"][phon_key] * confidence["snips"][fill_key]

        else:
            logging.error(f"sub_phrase {sub_phrase} is not supported")
            continue

        out_file = f"{index:05d}.wav"
        torchaudio.save(f"{out_dir}/{out_file}", wave_section, sr)

        log.log_time(ts_begin, int(index), df.shape[0], 100)

        info.append((
            row["filename"],
            phrase_confidence,
            duration
        ))

    confidence_df = pd.DataFrame(info, columns=["filename", "confidence", "duration_s"])

    df = pd.merge(df, confidence_df, on="filename")
    df.to_csv(os.path.join(out_dir, "info.csv"), index=False)

    if os.path.exists(os.path.join(source_dir, "parameter.json")):
        shutil.copyfile(
            os.path.join(source_dir, "parameter.json"),
            os.path.join(out_dir, "parameter.json")
        )
