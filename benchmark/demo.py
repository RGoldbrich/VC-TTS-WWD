# standard lib
import queue
import threading
import time
from collections import deque

# third party
import numpy as np
import sounddevice  # this is required to suppress ALSA LIB warnings
import torch
import torchaudio

# application
from auxiliary.recording import record_wave
from common.audio import normalize
from data.features import get_lfbe_transform
from model.classifier import get_master_model

segment_queue = queue.Queue()


def record_audio_thread():
    while True:
        w, _ = record_wave(300)
        segment_queue.put(w)


record_thread = threading.Thread(target=record_audio_thread)
record_thread.start()


def run_model():
    # load model
    model = get_master_model()
    model.load_state_dict(
        torch.load(
            "../model/out_train/25_02_03_Ablation/250209_151400_CRNN_98kp_70VC30TTS_241_21_LFBE_W25_H10_HailMary_9859_965tp_474fp/250209_154636_E40",
            # torch.load("../model/out_train/25_02_03_Ablation/250217_121129_CRNN_98kp_59VC41TTS_241_21_LFBE_W25_H10_SynthPart_9770_920tp_474fp/250217_122546_E16",
            map_location=torch.device('cpu')))

    model.eval()

    feature_transform = get_lfbe_transform(21)

    wave_segment_rotation = deque()
    pred_rotation = deque([0, 0, 0, 0, 0, 0, 0])

    t_feat, t_intf, t_rest = 0, 0, 0

    while True:
        if not segment_queue.empty():
            wave_segment_rotation.append(segment_queue.get())
            if len(wave_segment_rotation) == 8:
                # concatenate and predict
                t_before = time.time()
                raw_full = b''.join(wave_seg for wave_seg in wave_segment_rotation)

                # convert to float and torch tensor
                buffer = np.frombuffer(raw_full, dtype=np.int16).astype(np.float32) / 32767.0
                wave_to_predict = torch.frombuffer(buffer, dtype=torch.float32)

                t_rest += time.time() - t_before

                wave_to_predict, _ = normalize(wave_to_predict)
                wave_to_predict = wave_to_predict[None]

                t_before = time.time()
                features = feature_transform(wave_to_predict)
                if model.expects_rnn_shaped_features:
                    features = torch.permute(features, (0, 2, 1))
                t_feat += time.time() - t_before

                t_before = time.time()
                if model.expects_rnn_shaped_features:
                    pred = model(features)
                else:
                    pred = model(features[None])
                t_intf += time.time() - t_before

                t_before = time.time()
                pred = torch.nn.functional.sigmoid(pred).item()

                pred_rotation.append(pred)

                out_of_last_three = sum(1 for num in list(pred_rotation)[-3:] if num > .95)

                if pred > .95 and out_of_last_three > 2:
                    print(f"XXX [{min(pred, 0.9999):06.2%}] - WW detected")
                elif pred > .95:
                    print(f"XXX [{min(pred, 0.9999):06.2%}]")
                else:
                    print(f"    [{min(pred, 0.9999):06.2%}]")

                wave_segment_rotation.popleft()
                pred_rotation.popleft()
                t_rest += time.time() - t_before

        time.sleep(0.01)
