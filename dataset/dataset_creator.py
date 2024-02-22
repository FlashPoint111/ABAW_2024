import pandas as pd
import ffmpeg
from PIL import Image
import io
import numpy as np
import torch
from torchvision import transforms

_JPEG_HEADER = b"\xff\xd8"

def extract_frames(video_path: str,
                   fps: int = 10,
                   min_resize: int = 320):
    new_width = "if(gt(iw,ih),{},-1)".format(min_resize)
    new_height = "if(gt(iw,ih),-1,{})".format(min_resize)
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

def image_decode(frame_set: list):
    return list(map(lambda x: Image.open(io.BytesIO(x)), frame_set))

def img_process(img: Image, transform: transforms.Compose):
    return transform(img)

def extract_audio(video_path: str, sampling_rate: int = 16000):
    audio, _ = (
        ffmpeg.input(video_path)
        .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sampling_rate)
        .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True, quiet=True)
    )
    audio = np.frombuffer(audio, np.int16).flatten()
    return audio

def extract_label(label_path: str):
    with open(label_path, "r") as f:
        return list(map(lambda x: int(x.strip('\n')), f.readlines()[1:]))


if __name__ == '__main__':
    dataset = 'train'
    dataset = pd.read_csv(f'{dataset}.csv').to_dict(orient='records')
    
    transform = transforms.Compose([transforms.ToTensor()])
    audio_rate = 24000
    target_fps = 10
    audio_rate = 24000
    sequence_len = 32
    stride = 1

    count = 0
    output = []
    output_count = 1
    
    for path in dataset:
        print(path['video_path'])
        image = extract_frames(path['video_path'])
        audio = extract_audio(path['video_path'], audio_rate)
        label = extract_label(path['label_path'])
        fps = path['fps']
        for i in range(0, len(image) - 32, 1):
            imgs_sample = image[i: i+32]
            audio_sample = audio[i*(audio_rate // target_fps): (i+sequence_len)*(audio_rate // target_fps)]
            label_sample = label[round((i/target_fps)*fps) :  round(((i+sequence_len)/target_fps)*fps)]
            label_sample = np.average(np.array(label_sample, np.int8))
            output.append({"video": imgs_sample, "audio": audio_sample, "label": label_sample})
            count += 1
            if count == 16384:
                output_path = f'dataset_part_{output_count}.pth'
                torch.save(output, output_path)
                output_count += 1
                output = []
                count = 0
