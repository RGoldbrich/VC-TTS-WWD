# standard lib
import os
import random
import re


def generate_source_transcripts_from_libri_speech():
    libri_source_dir = "../Datasets/speech_voice/LibriSpeech/train-clean-100"
    transcript_word_target = 20

    txt_files = [os.path.join(root, file) for root, _, files in os.walk(libri_source_dir) for file in files if
                 file.endswith(".txt")]

    print("number of transcript files found:", len(txt_files))
    print("=================================================")

    transcript_list = []

    for txt_file in txt_files:
        transcript = ""
        with open(txt_file, "r") as file:
            for line in file:
                cleaned_line = line.strip()
                transcript += cleaned_line[cleaned_line.find(' ') + 1:] + " "

        transcript_list.append(transcript.lower().strip())

    n_chars, n_words = 0, 0
    for transcript in transcript_list:
        n_chars += len(transcript)
        n_words += transcript.count(' ') + 1

    print("number of words in transcripts:", n_words)
    print("number of chars in transcripts:", n_chars)
    print("=================================================")

    final_transcript_list = []

    for transcript in transcript_list:
        n_words = transcript.count(' ') + 1
        n_sections = round(n_words / transcript_word_target)

        transcript = ' ' + transcript + ' '
        sep_loc = [m.start() for m in re.finditer(' ', transcript)]

        for sec_idx in range(n_sections):
            begin = round(n_words / n_sections * sec_idx)
            end = round(n_words / n_sections * (sec_idx + 1))

            mid = (end - begin) // 2 + begin

            final_transcript = (transcript[sep_loc[begin]:sep_loc[mid]] + ". hey, snips." + transcript[
                sep_loc[mid]:sep_loc[
                    end]]).strip()
            final_transcript_list.append(final_transcript + '\n')

    random.shuffle(final_transcript_list)
    print("number of final transcripts:", len(final_transcript_list))

    with open("transcripts.txt", "w") as file:
        file.writelines(final_transcript_list)
