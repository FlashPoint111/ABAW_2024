import pandas as pd
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.randaugment import RandomAugment
import torch.backends.cudnn as cudnn
from dataset.video_process import custom_collate_fn

import utils.utils as utils
from dataset.video_process import ImageAudioDataset
from model.model import MultiModalModel
from optim import create_optimizer
from scheduler import create_scheduler
import time
import json
import datetime


def train(model, data_loader, optimizer, epoch, warmup_steps, device, scheduler):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad()
        image = sample["video"].to(device, non_blocking=True)
        audio = sample["audio"].to(device, non_blocking=True)
        label = sample["label"].to(device, non_blocking=True).long()
        if epoch > 0:
            alpha = 0.4
        else:
            alpha = 0.1

        loss = model(image, audio, label, alpha=alpha)
        loss.backward()
        optimizer.step()
        metric_logger.update(loss_ita=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
            # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    device = torch.device('cuda')

    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = 30
    warmup_steps = 1e-5

    print("Creating dataset")
    with open('./config/train.json') as f:
        configs = json.load(f)
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    pretrain_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            normalize,
        ])    

    dataset = ImageAudioDataset(configs, pretrain_transform)
    dataloader = DataLoader(dataset, batch_size=configs["batch_size"], collate_fn=custom_collate_fn)

    print("Creating model")
    model = MultiModalModel()
    model = model.to(device)

    arg_opt = utils.AttrDict({'opt': 'adamW', 'lr': 1e-4, 'weight_decay': 0.02})
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict({'sched': 'cosine', 'lr': 1e-4, 'epochs': 30, 'min_lr': 1e-5, 'decay_rate': 1, 'warmup_lr': 1e-5, 'warmup_epochs': 20, 'cooldown_epochs': 0})
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    checkpoint = None
    resume = False
    if checkpoint:
        checkpoint = torch.load(checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        if resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch']+1
        model.load_state_dict(state_dict)
        print('load checkpoint from %s'%checkpoint)

    model_without_ddp = model

    print("Start training")
    start_time = time.time()

    output_dir = './output'
    for epoch in range(start_epoch, max_epoch):

        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        train_stats = train(model, dataloader, optimizer, epoch, warmup_steps, device, lr_scheduler)
        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         }
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(output_dir, 'checkpoint_%02d.pth' % epoch))

            with open(os.path.join(output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


