# standard lib
import datetime
import json
import logging
import os
from time import time

OUT_DIR_STEM = "out"


def setup_logging(
        log_file: str,
        console_level: int = logging.DEBUG
) -> logging.FileHandler:
    logging_format = "%(asctime)s - %(levelname)s - %(message)s"

    # root logger to print all levels to console
    logging.basicConfig(level=console_level, format=logging_format, encoding='utf-8')

    # create logfile
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setFormatter(logging.Formatter(logging_format))
    handler.setLevel(logging.INFO)  # omit DEBUG level

    logging.getLogger().addHandler(handler)

    return handler


def get_or_create_parameter_json(directory: str):
    if os.path.exists(os.path.join(directory, "parameter.json")):
        with open(os.path.join(directory, "parameter.json"), "r") as f:
            parameter_json = json.load(f)
    else:
        parameter_json = dict()

    return parameter_json


def store_parameter_json(directory: str, parameter_json: dict):
    with open(f"{directory}/parameter.json", mode="w", encoding='utf-8') as file:
        json.dump(parameter_json, file)


def close_logging(
        handler: logging.FileHandler
) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()


def get_timestamp_str():
    return datetime.datetime.now().strftime("%y%m%d_%H%M%S")


def setup_out_dir(
        path: str,
        purpose: str,
        label: str = None
):
    major_out_dir = os.path.join(path, f"{OUT_DIR_STEM}_{purpose}")
    if not os.path.exists(major_out_dir):
        os.makedirs(major_out_dir)

    if label:
        minor_out_dir = os.path.join(major_out_dir, f"{get_timestamp_str()}_{label}")
    else:
        minor_out_dir = os.path.join(major_out_dir, f"{get_timestamp_str()}")
    os.makedirs(minor_out_dir)
    return minor_out_dir


def print_log(
        text: str,
) -> None:
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3] + " - CUSTOM - " + text)


def log_time(
        ts_begin: float,
        current: int,
        total: int,
        sparse_modulo: int = 1,
        print_instead: bool = False,
):
    if current % sparse_modulo == 0:
        elapsed_min = (time() - ts_begin) / 60
        pace = max(current, 1) / elapsed_min
        remaining_min = (total - current) / pace
        if print_instead:
            print_log(f"{current} / {total} ({current / total:.2%}) - {int(pace)} it/min - "
                      f"{int(elapsed_min)} min ela, {int(remaining_min)} min rem")
        else:
            logging.info(f"{current} / {total} ({current / total:.2%}) - {int(pace)} it/min - "
                         f"{int(elapsed_min)} min ela, {int(remaining_min)} min rem")
