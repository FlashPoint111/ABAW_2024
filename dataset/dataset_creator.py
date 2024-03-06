import json
import os

import ffmpeg
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


def extract_frames(image_path: str):
    image = []
    img_idx = []
    for root, dirs, files in os.walk(image_path):
        for file in files:
            if '.DS_Store' in file:
                continue
            image.append(Image.open(os.path.join(root, file)).convert('RGB'))
            img_idx.append(int(file.split('.')[0]))
    return image, sorted(img_idx)


def extract_audio(video_path: str, sampling_rate: int = 16000):
    audio, _ = (
        ffmpeg.input(video_path)
        .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sampling_rate)
        .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True, quiet=True)
    )
    audio = np.frombuffer(audio, np.int16).flatten().astype(np.float32) / 32768.0
    return audio


def extract_label(label_path: str):
    with open(label_path, "r") as f:
        return list(map(lambda x: int(x.strip('\n')), f.readlines()[1:]))


N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160


def mel_filters(n_mels: int = N_MELS) -> torch.Tensor:
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load("../pretrain/mel_filters.npz") as f:
        return torch.from_numpy(f[f"mel_{n_mels}"])


def log_mel_spectrogram(
        audio: np.ndarray,
        n_mels: int = 80
):
    window = torch.hann_window(N_FFT)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


if __name__ == '__main__':
    with open('../config/train_dataset.json') as f:
        config = json.load(f)
    dataset = pd.read_csv(config["dataset_path"]).to_dict(orient='records')

    transform = transforms.Compose([transforms.ToTensor()])
    sample_rate = config["sample_rate"]
    seq_len = config["seq_len"]
    stride = config["stride"]
    if not os.path.exists(config["output_path"]):
        os.makedirs(config["output_path"])
    output_count = 0

    for sample in dataset:
        print(sample['video_path'])
        image, img_idx = extract_frames(sample['image_path'])
        audio = extract_audio(sample['video_path'], sample_rate)
        total_audio = len(audio)
        label = extract_label(sample['label_path'])
        fps = sample['fps']
        total_frames = sample['total_frame']
        actual_frames = len(img_idx)
        i = 0
        imgs_sample = []
        audio_sample = []
        label_sample = []
        count = 0

        while i < actual_frames:
            if label[img_idx[i] - 1] == -1:
                i += 1
                continue
            imgs_sample.append(image[i])
            audio_sample.extend(audio[int(img_idx[i] * (total_audio // total_frames)): int(
                (img_idx[i] + 1) * (total_audio // total_frames) - 1)])
            label_sample.append(label[img_idx[i] - 1])
            count += 1
            if count == 32:
                audio_sample = log_mel_spectrogram(torch.Tensor(audio_sample))
                torch.save(({"video": imgs_sample, "audio": audio_sample, "label": label_sample}),
                           os.path.join(config["output_path"], f"{output_count}.pth"))
                output_count += 1
                count = 0
                imgs_sample = []
                audio_sample = []
                label_sample = []
                i += stride
            else:
                i += 1
