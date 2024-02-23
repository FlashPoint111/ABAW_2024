import pandas as pd
import ffmpeg
from PIL import Image
import io
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, IterableDataset
from utils.randaugment import RandomAugment
from typing import Optional, Union

_JPEG_HEADER = b"\xff\xd8"

class IterCreator(IterableDataset):
    def __init__(self, configs, transfrom):
        super().__init__()
        self.configs = configs
        self.transform = transfrom
        self.total_length = 0

    def __iter__(self):
        target_fps = 10
        audio_rate = 24000
        sequence_len = 32
        stride = 1
        for config in self.configs:
            video_path = config["video_path"]
            label_path = config["label_path"]
            fps = config["fps"]
            imgs_encoded = extract_frames(video_path, target_fps)
            audio = extract_audio(video_path, audio_rate)
            label = extract_label(label_path)
            imgs_decoded = image_decode(imgs_encoded)
            del imgs_encoded
            imgs_decoded = list(map(lambda x: img_process(x, self.transform), imgs_decoded))
            for i in range(0, len(imgs_decoded) - sequence_len, stride):
                imgs_sample = torch.stack(imgs_decoded[i: i+sequence_len], dim=0).permute(1, 0, 2, 3)
                audio_sample = torch.Tensor(audio[i*(audio_rate // target_fps): (i+sequence_len)*(audio_rate // target_fps)])
                audio_sample = log_mel_spectrogram(audio_sample)
                label_sample = label[round((i/target_fps)*fps) :  round(((i+sequence_len)/target_fps)*fps)]
                label_sample = np.average(np.array(label_sample, np.int8))
                sample = {"video": imgs_sample, "audio": audio_sample, "label": label_sample}
                self.total_length += 1
                yield sample



def img_process(img: Image, transform: transforms.Compose):
    return transform(img)


def extract_frames(video_path: str,
                   fps: int = 10,
                   min_resize: int = 224):
    new_width = "if(gt(iw,ih),{},-1)".format(min_resize)
    new_height = "if(gt(iw,ih),-1,224)".format(min_resize)
    cmd = (
        ffmpeg
        .input(video_path)
        .filter("fps", fps=fps)
        .filter("scale", new_width, new_height)
        .output("pipe:", format="image2pipe")
    )
    jpeg_bytes, _ = cmd.run(capture_stdout=True, quiet=True)
    jpeg_bytes = jpeg_bytes.split(_JPEG_HEADER)[1:]
    jpeg_bytes = map(lambda x: _JPEG_HEADER + x, jpeg_bytes)
    return list(jpeg_bytes)


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
        return list(map(lambda x: x.strip('\n'), f.readlines()[1:]))


def image_decode(frame_set: list):
    return list(map(lambda x: Image.open(io.BytesIO(x)), frame_set))



SAMPLE_RATE = 24000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160

def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load("../pretrain/mel_filters.npz") as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: np.ndarray,
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):

    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


if __name__ == '__main__':
    dataset = 'train'
    dataset = pd.read_csv(f'{dataset}.csv').values
    video_path = [{"video_path":dataset[0][0], "label_path":dataset[0][1], "fps":30}]
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    pretrain_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])    

    dataset = IterCreator(video_path, pretrain_transform)
    dataloader = DataLoader(dataset, batch_size=1)

    for data in dataloader:
        print(len(data))
