import csv
import time

import sounddevice  # this is required to suppress ALSA LIB warnings

import json
from common.audio import get_audio_files_by_dir, play_audio_file
from common.log import log_time

TAGS = ["Emotional", "VoiceAccent", "Pronunciation", "BadEnvironment", "UnintelligibleQuiet"]

DESC = {
    "Emotional": "Could you see TTS follow a similar timbre and tone?",
    "VoiceAccent": "Would you be surprised to hear this voice in a english native speech corpus like LibriSpeech?",
    "Pronunciation": "Is the wake word mispronounced e.g. 'Snaps', 'Snipes' or 'Sniiips' or 'He', 'ey'?",
    "BadEnvironment": "Is the equipment bad or environment noisy?",
    "UnintelligibleQuiet": "Is the audio very quiet or otherwise unintelligible?",
}


def get_single_tag(audio_file: str, tag: str) -> list:
    play_audio_file(audio_file)

    user_input = input(f"{tag}: ").strip()

    # repeat
    if not user_input:
        return get_single_tag(audio_file, tag)

    if user_input.lower() == "n":
        return []
    elif user_input.lower() == "y" or user_input.lower() == "1":
        return [TAGS[0]]

    return get_single_tag(audio_file, tag)


def get_tags(audio_file: str) -> list:
    play_audio_file(audio_file)

    user_input = input("tags: ").strip()

    # return when no tags provided
    if not user_input:
        return []

    # get list of tag ids
    try:

        tag_ids = [int(i.strip()) - 1 for i in user_input]
        tag_ids = sorted(tag_ids)
        tags = [TAGS[i] for i in tag_ids if 0 <= i <= len(TAGS)]

    except (ValueError, TypeError):
        print("Retry please")
        return get_tags(audio_file)

    return tags


def label_sonos(path: str, out_path: str) -> None:
    # load all audios
    audio_files = sorted(get_audio_files_by_dir(path))

    # start_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
    start_chars = ['3']

    audio_files = [a for a in audio_files if a.split('/')[-1][0] in start_chars]
    print("number of files:", len(audio_files))

    info = []

    ts_begin = time.time()

    # loop over audio files
    for idx, audio_file in enumerate(audio_files):
        for tag_id, tag in enumerate(TAGS):
            print(f"{tag_id + 1}: {tag} ({DESC[tag]})")

        print(audio_file)

        # tags = get_single_tag(audio_file, "Emotional")

        tags = get_tags(audio_file)
        # more_tags = [t for t in more_tags if t != "Emotional"]
        # tags.extend(more_tags)

        print(f"{tags}")

        info.append({
            "file_name": audio_file,
            "tags": tags,
        })

        # periodically save results
        if idx % 10 == 9:
            with open(f"{out_path}/tags.json", "w", encoding="utf-8") as f:
                json.dump(info, f)

        log_time(ts_begin, idx, len(audio_files), print_instead=True)

    # save when finished
    with open(f"{out_path}/tags.json", "w", encoding="utf-8") as f:
        json.dump(info, f)
