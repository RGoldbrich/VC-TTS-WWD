# standard lib
import logging
from enum import Enum
from time import time

# third party
import torch
import torchinfo
from torch import nn
from torch.utils.data import DataLoader

# application
from common.audio import build_feature_list
from model.dataset import EagerDataset
from model.architectures.classifier_components import CONVOUT_LSTMIN, build_lstm_stage, build_narrowing_cl_stage

DROPOUT_P_IN = 0.0
DROPOUT_P_CONV = 0.25
DROPOUT_P_RNN = 0.1
DROPOUT_P_LINEAR = 0.25


class ReshapeConvOutToLSTMIn(nn.Module):
    def __init__(self, ):
        super(ReshapeConvOutToLSTMIn, self).__init__()

    @staticmethod
    def forward(x):
        # squash channels and height into single dimension
        batch_size, channels, height, width = x.size()
        x = torch.reshape(x, (batch_size, channels * height, width))

        # swap feature and time dimension (needed for LSTM)
        return x.permute(0, 2, 1)


class LSTMModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModule, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=DROPOUT_P_RNN)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        x, _ = self.lstm(x, (h0, c0))
        return x[:, -1, :]


class GRUModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModule, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=DROPOUT_P_RNN)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        x, _ = self.gru.forward(x, h0)
        return x[:, -1, :]


class DebugPrint(nn.Module):
    def __init__(self):
        super(DebugPrint, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


class FeatureType(Enum):
    MFCC = 1
    LFBE = 2


class Classifier(nn.Module):
    def __init__(
            self,
            feature_type,
            expected_feature_size: torch.Size,
            expects_rnn_shaped_features: bool,
            network_architecture: list[dict],
    ):
        super(Classifier, self).__init__()

        self.feature_type = feature_type
        self.expected_feature_size = expected_feature_size
        self.expects_rnn_shaped_features = expects_rnn_shaped_features
        self.network_architecture = network_architecture

        self.architecture_stub = ""
        n_conv, n_max, n_ch_final_conv, rnn_n_layers, rnn_h_size = 0, 0, 0, 0, 0

        self.model = nn.Sequential()

        f_size = [expected_feature_size[-2], expected_feature_size[-1]]

        for idx, stage in enumerate(network_architecture):
            info = stage["info"]
            match stage["type"]:
                case "input-drop":
                    self.model.add_module(f"{idx}-input-drop", nn.Dropout(DROPOUT_P_IN))

                case "conv":
                    self.model.add_module(f"{idx}-conv2d",
                                          nn.Conv2d(info["in_channels"], info["out_channels"],
                                                    info["kernel_size"], info["stride"]))
                    self.model.add_module(f"{idx}-batch-norm", nn.BatchNorm2d(info["out_channels"]))
                    self.model.add_module(f"{idx}-activ", nn.LeakyReLU())
                    self.model.add_module(f"{idx}-dropout", nn.Dropout2d(DROPOUT_P_CONV))

                    n_conv += 1
                    n_ch_final_conv = info["out_channels"]

                    f_size[0] = (f_size[0] - (info["kernel_size"][0] - info["stride"][0])) // info["stride"][0]
                    f_size[1] = (f_size[1] - (info["kernel_size"][1] - info["stride"][1])) // info["stride"][1]

                    if self.architecture_stub == "":
                        self.architecture_stub += "C"

                case "max-pool":
                    self.model.add_module(f"{idx}-max-pool", nn.MaxPool2d(info["kernel_size"], info["stride"]))
                    n_max += 1

                    f_size[0] = (f_size[0] - (info["kernel_size"][0] - info["stride"][0])) // info["stride"][0]
                    f_size[1] = (f_size[1] - (info["kernel_size"][1] - info["stride"][1])) // info["stride"][1]

                case "reshape-conv-to-lstm":
                    self.model.add_module(f"{idx}-conv-to-lstm", ReshapeConvOutToLSTMIn())

                case "lstm":
                    self.model.add_module(f"{idx}-lstm",
                                          LSTMModule(info["input_size"], info["hidden_size"], info["num_layers"]))
                    self.model.add_module(f"{idx}-dropout", nn.Dropout(DROPOUT_P_RNN))

                    rnn_n_layers = info["num_layers"]
                    rnn_h_size = info["hidden_size"]

                    if self.architecture_stub == "" or self.architecture_stub.endswith("C"):
                        self.architecture_stub += "R"

                case "gru":
                    self.model.add_module(f"{idx}-gru",
                                          GRUModule(info["input_size"], info["hidden_size"], info["num_layers"]))
                    self.model.add_module(f"{idx}-dropout", nn.Dropout(DROPOUT_P_RNN))

                    rnn_n_layers = info["num_layers"]
                    rnn_h_size = info["hidden_size"]

                    if self.architecture_stub == "" or self.architecture_stub.endswith("C"):
                        self.architecture_stub += "R"

                case "linear":
                    self.model.add_module(f"{idx}-linear", nn.Linear(info["in_features"], info["out_features"]))
                    self.model.add_module(f"{idx}-layer-norm", nn.LayerNorm(info["out_features"]))
                    self.model.add_module(f"{idx}-activ", nn.LeakyReLU())
                    self.model.add_module(f"{idx}-dropout", nn.Dropout(DROPOUT_P_LINEAR))

                case "final":
                    self.model.add_module(f"{idx}-linear", nn.Linear(info["in_features"], 1))

                case "flatten":
                    self.model.add_module(f"{idx}-flatten", nn.Flatten())

                case "debug-print":
                    self.model.add_module(f"{idx}-flatten", DebugPrint())

                case _:
                    raise Exception(f"Unexpected stage type: {stage['type']}")

        self.architecture_stub += "NN"

        self.architecture_stub += f"_{sum(param.numel() for param in self.model.parameters() if param.requires_grad) // 1000}kp"

    def forward(self, x):
        return self.model.forward(x)


def get_master_model(larger: bool = False):
    fac = 2 if larger else 1
    return Classifier(FeatureType.LFBE, torch.Size([1, 1, 21, 241]), False,
                      [
                          {"type": "conv", "info": {
                              "in_channels": 1, "out_channels": 8 * fac,
                              "kernel_size": (4, 4), "stride": (1, 1),
                          }},
                          {"type": "max-pool", "info": {
                              "kernel_size": (2, 2), "stride": (2, 2),
                          }},
                          {"type": "conv", "info": {
                              "in_channels": 8 * fac, "out_channels": 16 * fac,
                              "kernel_size": (4, 5), "stride": (1, 2),
                          }},
                          {"type": "max-pool", "info": {
                              "kernel_size": (1, 2), "stride": (1, 2),
                          }},
                          {"type": "conv", "info": {
                              "in_channels": 16 * fac, "out_channels": 32 * fac,
                              "kernel_size": (3, 4), "stride": (1, 1),
                          }},
                      ] +
                      [CONVOUT_LSTMIN] +
                      build_lstm_stage(128 * fac, 128 if larger else 96, 1) +
                      build_narrowing_cl_stage(128 if larger else 96, 2))


def to_latex_table(stub, model_print, torchsum_print):
    arch = str(model_print).split('\n')[1:-1]
    para = torchsum_print.summary_list[2:]

    print("\t& & & " + str(para[0].input_size)[3:-1] + " \\\\ \\hline")

    # loop over
    layer_offset_due_to_lstm = 0
    for idx, line in enumerate(arch):
        if line == "  )" or "LSTMModule" in line:
            continue
        layer_type = line.strip().split(' ')[1].split('(')[0]
        ident = line.strip().split(' ')[0][1:-2]

        desc = line[line.find('(', 8) + 1:-1]
        if layer_type == "Conv2d":
            comma_loc = desc.find(',')
            desc = "in_channels=" + desc[0:comma_loc] + ", out_channels=" + desc[comma_loc + 2:]
        elif layer_type == "BatchNorm2d":
            desc = "channels=" + desc
        elif layer_type == "LSTM":
            comma_loc = desc.find(',')
            desc = "input_size=" + desc[0:comma_loc] + ", hidden_size=" + desc[comma_loc + 2:]
        elif layer_type == "LayerNorm":
            comma_loc = desc.find(',')
            desc = "features=" + desc[1:comma_loc] + desc[comma_loc + 2:]

        layer = para[idx + layer_offset_due_to_lstm]

        print("\t" + layer_type.replace("_", "\\_") + " & "
              + desc.replace("_", "\\_") + " & "
              + str(layer.num_params) + " & "
              + str(layer.output_size)[3:-1]
              + (" \\\\ \\hline \\hline" if layer_type in ["LeakyReLU", "Dropout2d", "MaxPool2d", "Flatten", "Dropout",
                                                           "LSTM"] else "\\\\ \\hline"))

        if ident == "lstm":
            layer_offset_due_to_lstm = -1
