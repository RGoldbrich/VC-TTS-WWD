# standard lib
import logging
import os
import pandas as pd
import torch
import zlib

# third party
from torch.utils.data import DataLoader, Dataset


class EagerDataset(Dataset):
    def __init__(
            self,
            srcs_labels: list[tuple[str, int]] = None,
            filters: list[tuple[float, float, [float, float], float]] = None,
            squeeze_transpose: bool = False,
            supress_log: bool = False,
    ):
        """
        load features from the specified directories into memory.

        :param srcs_labels: list of feature directories with associated labels (either 0 or 1)
        :param filters:
            list of tuples used to reduce the feature:
            ["confidence", "confidence_percentile", ["min_duration_s", "max_duration_s"], "target_fraction"]
        :param squeeze_transpose: transform from [batch, channel (1), feat, time] to [batch, time, feat] (used for RNNs)
        :param supress_log: supress log output
        """
        if not srcs_labels:
            return

        self.labels = []
        feature_list = []

        for idx, (feature_dir, label) in enumerate(srcs_labels):
            # gather features
            tensor_files = [name for name in os.listdir(feature_dir) if
                            name.endswith(".pt") and name.find("feat") >= 0]
            # gather alignment info
            alignment_files = [name for name in os.listdir(feature_dir) if
                               name.endswith(".pt") and name.find("alignment") >= 0]

            # SORT TENSOR FILES!
            tensor_files = sorted(tensor_files)
            # SORT ALIGNMENT FILES!
            alignment_files = sorted(alignment_files)

            # load info.csv
            df = pd.read_csv(os.path.join(feature_dir, "full_info.csv"))

            # load feature and alignment info
            features = torch.cat([torch.load(os.path.join(feature_dir, tf)) for tf in tensor_files])
            alignments = torch.cat([torch.load(os.path.join(feature_dir, af)) for af in alignment_files])

            # sort df and tensor lists by alignment
            df = df.sort_values(by=["alignment"])
            alignments, indices = torch.sort(alignments)
            features = features[indices]

            assert df.shape[0] == alignments.shape[0]
            assert df.shape[0] == features.shape[0]

            if filters:
                confidence_fl, confidence_percentile_fl, duration_fl, fraction_fl = filters[idx]
            else:
                confidence_fl, confidence_percentile_fl, duration_fl, fraction_fl = -1, -1, -1, -1

            if confidence_fl != -1:
                n_col = df.shape[0]

                mask = df["confidence"] >= confidence_fl
                indices = df.index[mask]
                df = df.iloc[indices]
                df.reset_index(drop=True, inplace=True)

                alignments = alignments[indices]
                features = features[indices]

                if not supress_log:
                    logging.info(
                        f"filtering for audios with raw confidence >= {confidence_fl} ({n_col} >> {df.shape[0]})")

            if duration_fl != -1:
                n_col = df.shape[0]

                mask = (duration_fl[0] <= df["duration_s"]) & (df["duration_s"] <= duration_fl[1])
                indices = df.index[mask]
                df = df.iloc[indices]
                df.reset_index(drop=True, inplace=True)

                alignments = alignments[indices]
                features = features[indices]

                if not supress_log:
                    logging.info(
                        f"filtering for audios with duration between {duration_fl[0]}s and {duration_fl[1]}s ({n_col} >> {df.shape[0]})")

            if fraction_fl != -1:
                n_col = df.shape[0]

                indices = torch.randperm(len(features))
                subset_indices = indices[:int(fraction_fl * len(features))]

                df = df.iloc[subset_indices]
                df.reset_index(drop=True, inplace=True)
                features = features[subset_indices]
                alignments = alignments[subset_indices]

                if not supress_log:
                    logging.info(f"forwarding fraction of {fraction_fl} ({n_col} >> {df.shape[0]})")

            # for f in features:
            #     checksum = zlib.crc32(f.contiguous().numpy().tobytes())
            #     print(checksum)

            if squeeze_transpose:
                # drop already included channel dimension and swap feature and time dimension
                # target: batch, time, feat
                features = torch.squeeze(features).transpose(1, 2)
            else:
                # target: batch, channel (1), feat, time
                pass

            feature_list.append(features)
            self.labels += [float(label)] * features.shape[0]

            if not supress_log:
                logging.info(f"{features.shape[0]} selected/filtered in {feature_dir}")
                logging.info("-----------------------------------------------------------")

        self.features = torch.cat(feature_list)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

        theoretical_size = self.features.numel() * 4 / (1024 ** 3)
        if not supress_log:
            logging.info(f"total number of features: {len(self.labels)} ({round(theoretical_size, 2)} GiB)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.features[item], self.labels[item]

    def get_label_weights(self) -> tuple[int, int]:
        pos_count = self.labels.count_nonzero().item()
        return len(self.labels) - pos_count, pos_count
