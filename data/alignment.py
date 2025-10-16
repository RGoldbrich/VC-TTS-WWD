# standard lib
import logging
import os.path

# third party
import json
import matplotlib.pyplot as plt
import torch
import torchaudio
import pandas as pd
from time import time

# application
from common.audio import get_audio_files_by_dir
import common.log as log


def clean_transcript(transcript: str):
    return transcript.replace('.', '').replace(',', '')


def plot_emission(emission):
    fig, ax = plt.subplots()
    ax.imshow(emission.cpu().T)
    ax.set_title("Frame-wise class probabilities")
    ax.set_xlabel("Time")
    ax.set_ylabel("Labels")
    fig.tight_layout()
    plt.show()


def align(emission, tokens, device):
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = torchaudio.functional.forced_align(emission, targets, blank=0)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability
    return alignments, scores


def unflatten(list_, lengths):
    assert len(list_) == sum(lengths)

    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i: i + l])
        i += l
    return ret


def phrase_confidence(span, alignment_scores):
    # compute confidence in phonemes using (averaged if multi span token)
    phoneme_confidence = torch.Tensor([t.score for t in span])

    # confidence of empty frames between the phonemes
    filler_confidence = torch.cat(
        [alignment_scores[span[t].end:span[t + 1].start] for t in range(len(span) - 1)]).to("cpu")

    # time_conf_debug = alignment_scores[span[0].start: span[-1].end].to("cpu")

    return {
        "phoneme_ari_mean": phoneme_confidence.mean().item(),
        "phoneme_gen_mean_p2": (phoneme_confidence.pow(2).sum() / phoneme_confidence.shape[0]).pow(1 / 2).item(),
        "phoneme_gen_mean_p3": (phoneme_confidence.pow(3).sum() / phoneme_confidence.shape[0]).pow(1 / 3).item(),
        "phoneme_gen_mean_p4": (phoneme_confidence.pow(4).sum() / phoneme_confidence.shape[0]).pow(1 / 4).item(),
        "filler_ari_mean": filler_confidence.mean().item(),
        "filler_gen_mean_p-1": (filler_confidence.pow(-1).sum() / filler_confidence.shape[0]).pow((1 / -1)).item(),
        "filler_gen_mean_p-2": (filler_confidence.pow(-2).sum() / filler_confidence.shape[0]).pow((1 / -2)).item(),
        "filler_gen_mean_p-3": (filler_confidence.pow(-3).sum() / filler_confidence.shape[0]).pow((1 / -3)).item(),
    }


def align_single_audio(audio_path: str, transcript: list[str], model, LABELS, DICTIONARY, device):
    wave, sr = torchaudio.load(audio_path)

    with torch.inference_mode():
        emission, _ = model(wave.to(device))

    tokenized_transcript = [DICTIONARY[c] for word in transcript for c in word]

    aligned_tokens, alignment_scores = align(emission, tokenized_transcript, device)
    # for i, (ali, score) in enumerate(zip(aligned_tokens, alignment_scores)):
    #     print(f"{i:3d}:\t{ali:2d} [{LABELS[ali]}], {score:.2f}")

    token_spans = torchaudio.functional.merge_tokens(aligned_tokens, alignment_scores)

    word_spans = unflatten(token_spans, [len(word) for word in transcript])
    assert len(transcript) == len(word_spans), "word span and transcript mismatch"

    ratio = wave.shape[-1] / len(aligned_tokens)

    # word-level alignment
    word_level_alignment = []
    for idx, word in enumerate(word_spans):
        begin_frame = int(word[0].start * ratio)
        end_frame = int(word[-1].end * ratio)

        word_level_alignment.append((begin_frame / sr, end_frame / sr))

    hey_index = transcript.index("hey")
    snips_index = transcript.index("snips")
    assert hey_index + 1 == snips_index, "hey and snips position"

    # confidence in "hey"
    hey_span = word_spans[hey_index]
    hey_confidence = phrase_confidence(hey_span, alignment_scores)

    # confidence in "snips"
    snips_span = word_spans[snips_index]
    snips_confidence = phrase_confidence(snips_span, alignment_scores)

    # confidence of "hey snips"
    hey_span.extend(snips_span)
    hey_snips_confidence = phrase_confidence(hey_span, alignment_scores)

    target_phrase_confidence = {
        "hey": hey_confidence,
        "snips": snips_confidence,
        "hey snips": hey_snips_confidence,
    }

    target_grapheme_confidence = [t.score for t in hey_span]

    return word_level_alignment, target_phrase_confidence, target_grapheme_confidence


def align_dataset(source_dir: str) -> None:
    log_handler = log.setup_logging(source_dir + "/align.log")
    logging.info("start of align_dataset")

    df = pd.read_csv(source_dir + "/info.csv")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=False).to(device)

    LABELS = bundle.get_labels(star=None)
    DICTIONARY = bundle.get_dict(star=None)

    alignment_info = dict()
    phrase_info = dict()
    grapheme_info = dict()

    ts_begin = time()

    n_failing_alignment = 0

    logging.info(f"Audio files in csv before alignment: {df.shape[0]}")

    for index, row in df.iterrows():
        logging.info(f"Processing audio file: {row['filename']}")

        transcript = clean_transcript(row["transcript"]).split()

        try:
            word_alignment, hey_snips_aux, grapheme_scores = align_single_audio(
                os.path.join(source_dir, row["filename"]),
                transcript,
                model, LABELS, DICTIONARY, device
            )
            alignment_info[row['filename']] = word_alignment
            phrase_info[row['filename']] = hey_snips_aux
            grapheme_info[row['filename']] = grapheme_scores

        except Exception as e:
            logging.warning(f"{os.path.join(source_dir, row['filename'])}: {e} (removing from source)")
            if os.path.exists(os.path.join(source_dir, row["filename"])):
                os.remove(os.path.join(source_dir, row["filename"]))
            n_failing_alignment += 1

        log.log_time(ts_begin, int(index), df.shape[0], 25)

    with open(os.path.join(source_dir, "alignment.json"), "w") as file:
        json.dump(alignment_info, file)

    with open(os.path.join(source_dir, "confidence.json"), "w") as file:
        json.dump(phrase_info, file)

    with open(os.path.join(source_dir, "grapheme.json"), "w") as file:
        json.dump(grapheme_info, file)

    logging.info(f"{n_failing_alignment} audios failed alignment; down to {df.shape[0] - n_failing_alignment}")
    log.close_logging(log_handler)


def delete_missing_from_info_csv(source_dir: str) -> None:
    df = pd.read_csv(source_dir + "/info.csv")
    actual_files = get_audio_files_by_dir(source_dir)
    actual_files = [f.split('/')[-1] for f in actual_files]

    supposed_files = df["filename"]
    mask = [True if f in actual_files else False for f in supposed_files]

    df = df[mask]
    df.reset_index(drop=True, inplace=True)
    df.to_csv(source_dir + "/info_new.csv", index=False)


def create_info_csv():
    filenames = [f.split('/')[-1] for f in get_audio_files_by_dir("../Datasets/speech_voice/Snips/pos_train")]
    transcripts = ["hey snips"] * len(filenames)

    info = {
        "filename": filenames,
        "transcript": transcripts,
    }

    pd.DataFrame(info).to_csv("../Datasets/speech_voice/Snips/pos_train/info.csv", index=False)
