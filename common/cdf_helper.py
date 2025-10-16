import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from data.alignment import clean_transcript


class SubPhraseCdf:
    def __init__(self, source_csv_path: str, alignment_info_path: str, sub_phrase: str):
        assert sub_phrase in ["hey snips", "hey", "snips", "hey _", "_ snips", "front", "back"]

        self.df = pd.read_csv(source_csv_path)
        n = self.df.shape[0]

        with open(alignment_info_path, 'r') as f:
            alignment_info = json.load(f)

        # compute min mean confidence scores
        min_mean_confidence_scores = []
        for index, row in self.df.iterrows():
            alignment = alignment_info[row["filename"]]
            transcript = clean_transcript(row["transcript"]).split()

            assert len(alignment) == len(transcript)

            hey_index = transcript.index("hey")

            cs_indx = 5

            score = 0
            if sub_phrase == "hey snips":
                score = min(
                    alignment[hey_index][cs_indx],
                    alignment[hey_index + 1][cs_indx]
                )
            elif sub_phrase == "hey":
                score = alignment[hey_index][cs_indx]
            elif sub_phrase == "snips":
                score = alignment[hey_index + 1][cs_indx]
            elif sub_phrase == "hey _":
                score = min(
                    alignment[hey_index][cs_indx],
                    alignment[hey_index + 2][cs_indx],
                    alignment[hey_index + 3][cs_indx]
                )
            elif sub_phrase == "_ snips":
                score = min(
                    alignment[hey_index - 2][cs_indx],
                    alignment[hey_index - 1][cs_indx],
                    alignment[hey_index + 1][cs_indx]
                )
            elif sub_phrase == "front":
                score = alignment[hey_index][cs_indx]  # use confidence in placing the "hey"
            elif sub_phrase == "back":
                score = alignment[hey_index + 1][2]  # use confidence in placing the "snips"

            min_mean_confidence_scores.append(score)

        self.df["min_mean_confidence_score"] = min_mean_confidence_scores
        self.df = self.df.sort_values(by="min_mean_confidence_score", ignore_index=True)

        self.df["percentile"] = np.arange(n) / n

    def get_confidence_percentile(self, filename: str):
        return (self.df[self.df["filename"] == filename]["min_mean_confidence_score"].item(),
                self.df[self.df["filename"] == filename]["percentile"].item())


class BetterVolCdfUtil:
    def __init__(self, source_csv_path: str, target_csv_path: str, column: str):
        # 1:    source csv
        df = pd.read_csv(source_csv_path)
        n = df.shape[0]
        # print("number of rows in source:", n)
        df = df[["prepare_filename", column]]  # filter for relevant columns
        df = df.sort_values([column], ignore_index=True)  # sort by specified column

        # add cdf value to df
        source_cdf = np.arange(n) / n
        df["cdf"] = source_cdf

        # save in instance
        self.source_cdf_step = 1 / n
        self.source_df = df

        # 2:    target csv
        df = pd.read_csv(target_csv_path)
        n = df.shape[0]
        # print("number of rows in target:", n)
        df = df[["filename", column]]  # filter for relevant columns
        df = df.sort_values([column], ignore_index=True)  # sort by specified column

        # add cdf value to df
        target_cdf = np.arange(n) / n
        df["cdf"] = target_cdf

        # interpolate target volume levels at cdf-values from source
        self.target_volumes = np.interp(source_cdf, target_cdf, df[column])
        # print(self.target_volumes)
        # print("number of target volumes", len(self.target_volumes))

    def convert_vol(self, prepare_filename: str):
        cdf_in_source = self.source_df[self.source_df["prepare_filename"] == prepare_filename]["cdf"].item()

        # print("cdf_in_source", cdf_in_source)

        index = int(cdf_in_source / self.source_cdf_step)
        # print("index in target volumes", index)

        target_vol = self.target_volumes[index]
        # print("target volume", target_vol)
        return target_vol
