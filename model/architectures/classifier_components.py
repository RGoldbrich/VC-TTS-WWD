# standard lib
import math

# third party
import torch

INPUT_DROPOUT = {"type": "input-drop", "info": None}
DEBUG_PRINT = {"type": "debug-print", "info": None}

FLATTENING = {"type": "flatten", "info": None}
CONVOUT_LSTMIN = {"type": "reshape-conv-to-lstm", "info": None}

FULL_CONVOLUTIONAL_STAGES = {
    # baseline cnn; 4 conv with 2 pool
    "baseline": {
        "input-size": torch.Size([1, 1, 13, 121]),
        "output-size": torch.Size([1, 64, 2, 4]),
        "model": [
            {"type": "conv", "info": {
                "in_channels": 1, "out_channels": 16,
                "kernel_size": (3, 4), "stride": (1, 1),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (1, 2), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 16, "out_channels": 32,
                "kernel_size": (3, 5), "stride": (1, 2),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (1, 2), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 32, "out_channels": 64,
                "kernel_size": (4, 6), "stride": (1, 1),
            }},
            {"type": "conv", "info": {
                "in_channels": 64, "out_channels": 64,
                "kernel_size": (5, 6), "stride": (1, 1),
            }},
        ]
    },
    "deep": {  # more channels
        "input-size": torch.Size([1, 1, 13, 121]),
        "output-size": torch.Size([1, 128, 2, 4]),
        "model": [
            {"type": "conv", "info": {
                "in_channels": 1, "out_channels": 32,
                "kernel_size": (3, 4), "stride": (1, 1),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (1, 2), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 32, "out_channels": 64,
                "kernel_size": (3, 5), "stride": (1, 2),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (1, 2), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 64, "out_channels": 128,
                "kernel_size": (4, 6), "stride": (1, 1),
            }},
            {"type": "conv", "info": {
                "in_channels": 128, "out_channels": 128,
                "kernel_size": (5, 6), "stride": (1, 1),
            }},
        ]
    },
    "shallow": {  # less channels
        "input-size": torch.Size([1, 1, 13, 121]),
        "output-size": torch.Size([1, 32, 2, 4]),
        "model": [
            {"type": "conv", "info": {
                "in_channels": 1, "out_channels": 8,
                "kernel_size": (3, 4), "stride": (1, 1),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (1, 2), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 8, "out_channels": 16,
                "kernel_size": (3, 5), "stride": (1, 2),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (1, 2), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 16, "out_channels": 32,
                "kernel_size": (4, 6), "stride": (1, 1),
            }},
            {"type": "conv", "info": {
                "in_channels": 32, "out_channels": 32,
                "kernel_size": (5, 6), "stride": (1, 1),
            }},
        ]
    },

    # more aggressive models; 3 conv with 2 pool
    "aggressive": {
        "input-size": torch.Size([1, 1, 13, 121]),
        "output-size": torch.Size([1, 64, 2, 4]),
        "model": [
            {"type": "conv", "info": {
                "in_channels": 1, "out_channels": 16,
                "kernel_size": (3, 4), "stride": (1, 1),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (1, 2), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 16, "out_channels": 32,
                "kernel_size": (4, 5), "stride": (1, 2),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (1, 2), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 32, "out_channels": 64,
                "kernel_size": (6, 8), "stride": (2, 2),
            }},
        ]
    },
    "aggressive-deep": {  # more channels
        "input-size": torch.Size([1, 1, 13, 121]),
        "output-size": torch.Size([1, 128, 2, 4]),
        "model": [
            {"type": "conv", "info": {
                "in_channels": 1, "out_channels": 32,
                "kernel_size": (3, 4), "stride": (1, 1),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (1, 2), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 32, "out_channels": 64,
                "kernel_size": (4, 5), "stride": (1, 2),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (1, 2), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 64, "out_channels": 128,
                "kernel_size": (6, 8), "stride": (2, 2),
            }},
        ]
    },

    # violent models; 3 conv with 1 pool
    "violent": {
        "input-size": torch.Size([1, 1, 13, 121]),
        "output-size": torch.Size([1, 64, 2, 4]),
        "model": [
            {"type": "conv", "info": {
                "in_channels": 1, "out_channels": 16,
                "kernel_size": (5, 7), "stride": (2, 2),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (1, 2), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 16, "out_channels": 32,
                "kernel_size": (3, 7), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 32, "out_channels": 64,
                "kernel_size": (3, 6), "stride": (1, 2),
            }},
        ]
    },
    "violent-deep": {  # more channels
        "input-size": torch.Size([1, 1, 13, 121]),
        "output-size": torch.Size([1, 128, 2, 4]),
        "model": [
            {"type": "conv", "info": {
                "in_channels": 1, "out_channels": 32,
                "kernel_size": (5, 7), "stride": (2, 2),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (1, 2), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 32, "out_channels": 64,
                "kernel_size": (3, 7), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 64, "out_channels": 128,
                "kernel_size": (3, 6), "stride": (1, 2),
            }},
        ]
    },
}

PARTIAL_CONVOLUTIONAL_STAGES = {
    # baseline; derived from best performing cnn
    "baseline": {
        "input-size": torch.Size([1, 1, 13, 121]),
        "output-size": torch.Size([1, 16, 8, 28]),
        "model": [
            {"type": "conv", "info": {
                "in_channels": 1, "out_channels": 8,
                "kernel_size": (3, 4), "stride": (1, 1),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (1, 2), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 8, "out_channels": 16,
                "kernel_size": (4, 5), "stride": (1, 2),
            }},
        ]
    },
    "time-plus": {  # baseline + more time res
        "input-size": torch.Size([1, 1, 13, 121]),
        "output-size": torch.Size([1, 16, 8, 55]),
        "model": [
            {"type": "conv", "info": {
                "in_channels": 1, "out_channels": 8,
                "kernel_size": (3, 4), "stride": (1, 1),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (1, 2), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 8, "out_channels": 16,
                "kernel_size": (4, 5), "stride": (1, 1),
            }},
        ]
    },
    "time-minus": {  # baseline + less time res
        "input-size": torch.Size([1, 1, 13, 121]),
        "output-size": torch.Size([1, 16, 8, 13]),
        "model": [
            {"type": "conv", "info": {
                "in_channels": 1, "out_channels": 8,
                "kernel_size": (3, 7), "stride": (1, 2),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (1, 2), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 8, "out_channels": 16,
                "kernel_size": (4, 5), "stride": (1, 2),
            }},
        ]
    },

    # deeper conv (req. less freq domain and more channel)
    # note: best performing
    "aggressive": {
        "input-size": torch.Size([1, 1, 13, 121]),
        "output-size": torch.Size([1, 32, 4, 24]),
        "model": [
            {"type": "conv", "info": {
                "in_channels": 1, "out_channels": 8,
                "kernel_size": (3, 4), "stride": (1, 1),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (1, 2), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 8, "out_channels": 16,
                "kernel_size": (4, 5), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 16, "out_channels": 32,
                "kernel_size": (5, 5), "stride": (1, 1),
            }},
        ]
    },
    "aggressive-pref-ch": {  # more channel; less freq
        "input-size": torch.Size([1, 1, 13, 121]),
        "output-size": torch.Size([1, 64, 2, 24]),
        "model": [
            {"type": "conv", "info": {
                "in_channels": 1, "out_channels": 16,
                "kernel_size": (3, 4), "stride": (1, 1),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (1, 2), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 16, "out_channels": 32,
                "kernel_size": (5, 5), "stride": (2, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 32, "out_channels": 64,
                "kernel_size": (3, 5), "stride": (1, 1),
            }},
        ]
    },
    "aggressive-ch-fr-extra-256": {  # more channels, same freq (more complex and worse)
        "input-size": torch.Size([1, 1, 13, 121]),
        "output-size": torch.Size([1, 64, 4, 24]),
        "model": [
            {"type": "conv", "info": {
                "in_channels": 1, "out_channels": 16,
                "kernel_size": (3, 4), "stride": (1, 1),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (1, 2), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 16, "out_channels": 32,
                "kernel_size": (4, 5), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 32, "out_channels": 64,
                "kernel_size": (5, 5), "stride": (1, 1),
            }},
        ]
    },

    # adaptations of best performing c-stage for different frequency res
    "aggressive-adap21": {
        "input-size": torch.Size([1, 1, 21, 121]),
        "output-size": torch.Size([1, 32, 4, 24]),
        "model": [
            {"type": "conv", "info": {
                "in_channels": 1, "out_channels": 8,
                "kernel_size": (4, 4), "stride": (1, 1),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (1, 2), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 8, "out_channels": 16,
                "kernel_size": (4, 5), "stride": (2, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 16, "out_channels": 32,
                "kernel_size": (5, 5), "stride": (1, 1),
            }},
        ]
    },
    "aggressive-adap25": {
        "input-size": torch.Size([1, 1, 25, 121]),
        "output-size": torch.Size([1, 32, 4, 24]),
        "model": [
            {"type": "conv", "info": {
                "in_channels": 1, "out_channels": 8,
                "kernel_size": (4, 4), "stride": (1, 1),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (2, 2), "stride": (2, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 8, "out_channels": 16,
                "kernel_size": (5, 5), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 16, "out_channels": 32,
                "kernel_size": (4, 5), "stride": (1, 1),
            }},
        ]
    },
    "aggressive-adap33": {
        "input-size": torch.Size([1, 1, 33, 121]),
        "output-size": torch.Size([1, 32, 4, 24]),
        "model": [
            {"type": "conv", "info": {
                "in_channels": 1, "out_channels": 8,
                "kernel_size": (4, 4), "stride": (1, 1),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (2, 2), "stride": (2, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 8, "out_channels": 16,
                "kernel_size": (5, 5), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 16, "out_channels": 32,
                "kernel_size": (5, 5), "stride": (2, 1),
            }},
        ]
    },
    "aggressive-adap33-B": {
        "input-size": torch.Size([1, 1, 33, 121]),
        "output-size": torch.Size([1, 32, 4, 24]),
        "model": [
            {"type": "conv", "info": {
                "in_channels": 1, "out_channels": 8,
                "kernel_size": (4, 4), "stride": (1, 1),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (1, 2), "stride": (1, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 8, "out_channels": 16,
                "kernel_size": (6, 5), "stride": (2, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 16, "out_channels": 32,
                "kernel_size": (7, 5), "stride": (2, 1),
            }},
        ]
    },
    "aggressive-adap40": {
        "input-size": torch.Size([1, 1, 40, 121]),
        "output-size": torch.Size([1, 32, 4, 24]),
        "model": [
            {"type": "conv", "info": {
                "in_channels": 1, "out_channels": 8,
                "kernel_size": (3, 4), "stride": (1, 1),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (2, 2), "stride": (2, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 8, "out_channels": 16,
                "kernel_size": (5, 5), "stride": (2, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 16, "out_channels": 32,
                "kernel_size": (5, 5), "stride": (1, 1),
            }},
        ]
    },
    "aggressive-adap64": {
        "input-size": torch.Size([1, 1, 64, 121]),
        "output-size": torch.Size([1, 32, 4, 24]),
        "model": [
            {"type": "conv", "info": {
                "in_channels": 1, "out_channels": 8,
                "kernel_size": (3, 4), "stride": (1, 1),
            }},
            {"type": "max-pool", "info": {
                "kernel_size": (2, 2), "stride": (2, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 8, "out_channels": 16,
                "kernel_size": (7, 5), "stride": (2, 2),
            }},
            {"type": "conv", "info": {
                "in_channels": 16, "out_channels": 32,
                "kernel_size": (7, 5), "stride": (2, 1),
            }},
        ]
    },
}


def build_lstm_stage(input_size: int, hidden_size: int, num_layers: int):
    return [{"type": "lstm", "info": {"input_size": input_size, "hidden_size": hidden_size, "num_layers": num_layers}}]


def build_gru_stage(input_size: int, hidden_size: int, num_layers: int):
    return [{"type": "gru", "info": {"input_size": input_size, "hidden_size": hidden_size, "num_layers": num_layers}}]


def build_narrowing_cl_stage(input_size: int, num_layers: int):
    if num_layers == 1:
        return build_cl_stage_raw([input_size])

    layer_sizes = [input_size]
    cur_size_log = int(math.log2(input_size))

    for n in range(num_layers - 1):
        cur_size_log -= 1
        assert cur_size_log > 0
        layer_sizes.append(2 ** cur_size_log)

    return build_cl_stage_raw(layer_sizes)


def build_cl_stage_raw(size: list[int]):
    clf_stage = []

    for i in range(1, len(size)):
        clf_stage.append({"type": "linear", "info": {"in_features": size[i - 1], "out_features": size[i]}})

    clf_stage.append({"type": "final", "info": {"in_features": size[-1]}})
    return clf_stage


if __name__ == "__main__":
    print(build_narrowing_cl_stage(32, 6))
