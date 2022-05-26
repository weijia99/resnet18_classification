# 1.首先是main函数的构建

# 判断是不是train的模式
import argparse
import glob
import math
import sys
from collections.abc import Iterable
from pathlib import Path

import os
import timm

import torch.utils.data
import torchvision.transforms
from PIL import Image
from timm.utils import accuracy
from torch.utils.tensorboard import SummaryWriter
from misc import NativeScalerWithGradNormCount as NativeScaler
import misc


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=72, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory '
                             'constraints)')

    # Model parameters
    # parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
    #                     help='Name of model to train')

    parser.add_argument('--input_size', default=128, type=int,
                        help='images input size')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (absolute lr)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--root_path', default='./', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')

    # 自定义的模型加载
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--pin_memory', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transform(is_train, args):
    if is_train:
        # 进行训练集的transform，设置
        print("train transform")
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize((args.input_size, args.input_size)),
             torchvision.transforms.RandomHorizontalFlip(),
             torchvision.transforms.RandomVerticalFlip(),
             torchvision.transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
             torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
             torchvision.transforms.ToTensor()]

        )
    else:
        print("eval mode")
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize((args.input_size, args.input_size)),
             torchvision.transforms.ToTensor()]

        )
    return transform


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    path = os.path.join(args.root_path, 'train1' if is_train else 'test1')
    dataset = torchvision.datasets.ImageFolder(path, transform=transform)
    info = dataset.find_classes(path)
    print(f"finding classes from {path} :\t{info[0]}")
    print(f'mapping class from {path},\t{info[1]}')
    return dataset


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter=" ")
    header = "Test:"

    # 转入到评测模式
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # 计算loss，统计得分
        with torch.no_grad():
            output = model(images)
            loss = criterion(output, target)

        output = torch.nn.functional.softmax(output, dim=1)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        # 写下在tensorboard
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print(f'acc1:{metric_logger.acc1}, acc5:{metric_logger.acc5}')
    # return {k: meter.global_avg for k, meter in metric_logger.items()}
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, log_writer=None,
                    args=None):
    if log_writer is not None:
        print(f'log_dir {log_writer.log_dir}')
    model.train(True)
    print_feq = 2
    accum_iter = args.accum_iter

    for data_iter_step, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        output = model(samples)

        warmup_lr = args.lr
        optimizer.param_groups[0]["lr"] = warmup_lr

        loss = criterion(output, targets)
        loss /= accum_iter

        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        loss_val = loss.item()

        # 完成一次epoch就进行置为，走
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if not math.isfinite(loss_val):
            print(f"loss is {loss_val} & stop training")
            sys.exit(1)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss", loss, epoch_1000x)
            log_writer.add_scalar("lr", warmup_lr, epoch_1000x)
            print(f'Epoch:{epoch},loss:{loss},step:{data_iter_step},lr:{warmup_lr}')

    # # gather the stats from all processes
    # log_writer.synchronize_between_processes()
    # print("Averaged stats:", log_writer)
    # return {k: meter.global_avg for k, meter in log_writer.meters.items()}


def main(args, mode='train', test_img_path=''):
    print(f'{mode} is loading')
    if mode == 'train':
        dataset_train = build_dataset(is_train=True, args=args)
        dataset_val = build_dataset(is_train=False, args=args)
        # 接下来就设置如何抽取，train是random，test就是随机

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        # 接下来就是钢构件自己的dataloader

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            drop_last=True
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            drop_last=False
        )

        # 2.构建model，这里我们可以用timm的直接构建,设置一些参数

        model = timm.create_model("resnet18", pretrained=False, num_classes=36,drop_rate=0.1, drop_path_rate=0.1)
        model.to(device)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # 计算当前一共有多是个参数量

        print('number of parameters: %.2f' % (n_parameters / 1.e6))

        # 3.loss构建
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # 4.构建Tensorboard
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
        loss_scaler = NativeScaler()
        # 4.1导入何开明的加载模型函数
        misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)
        # 5.训练教程

        for epoch in range(args.start_epoch, args.epochs):
            print(f'Epoch {epoch}')
            print(f'length of dataloader_train is {len(data_loader_train)}')

            if epoch % 1 == 0:
                # 这个是评估过程,使用的验证集
                print("Evaluating ...")
                model.eval()
                test_stats = evaluate(data_loader_val, model, device)
                print(f"accuracy  of network on the {len(dataset_val)} test image:{test_stats['acc1']:.1f}%")
                if log_writer is not None:
                    log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                    log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
                    log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

                # 移动到训练模式
            model.train()
            train_stats = train_one_epoch(
                    model, criterion, data_loader_train,
                    optimizer, device, epoch + 1,
                    loss_scaler, None,
                    log_writer=log_writer,
                    args=args
                )

                # save model
            if args.output_dir:
                    print("saving checkpoints....")
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model,
                        optimizer=optimizer, epoch=epoch, loss_scaler=loss_scaler
                    )
    else:
        # 走入训练模式,加载我们的模型
        model = timm.create_model("resnet18", pretrained=False,
                                  drop_rate=0.1, drop_path_rate=0.1)
        # model.to(device)

        classes_dir = {'apple': 0, 'banana': 1, 'beetroot': 2, 'bell pepper': 3, 'cabbage': 4, 'capsicum': 5,
                       'carrot': 6, 'cauliflower': 7, 'chilli pepper': 8, 'corn': 9, 'cucumber': 10, 'eggplant': 11,
                       'garlic': 12, 'ginger': 13, 'grapes': 14, 'jalepeno': 15, 'kiwi': 16, 'lemon': 17, 'lettuce': 18,
                       'mango': 19, 'onion': 20, 'orange': 21, 'paprika': 22, 'pear': 23, 'peas': 24, 'pineapple': 25,
                       'pomegranate': 26, 'potato': 27, 'raddish': 28, 'soy beans': 29, 'spinach': 30, 'sweetcorn': 31,
                       'sweetpotato': 32, 'tomato': 33, 'turnip': 34, 'watermelon': 35}
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of parameters: %.2f' % (n_parameters / 1.e6))

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # 4.构建Tensorboard
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
        loss_scaler = NativeScaler()

        # 5.加载 训练的模型来预测，通过这个来获取我们pth来得到答案
        misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer,
                        loss_scaler=loss_scaler)
        inchannel = model.fc.in_features
        model.fc = torch.nn.Linear(inchannel, 36)
        model.eval()

        # 传入要预测的模型
        image = Image.open(test_img_path).convert('RGB')
        image = image.resize((args.input_size, args.input_size), Image.ANTIALIAS)
        image = torchvision.transforms.ToTensor()(image).unsqueeze(0)

        # 进行推理
        with torch.no_grad():
            output = model(image)

        output = torch.nn.functional.softmax(output, dim=-1)
        class_idx = torch.argmax(output, dim=1)[0]
        score = torch.max(output, dim=1)[0][0]
        print(f'image path is {test_img_path}')
        print(
            f'score is {score.item()},class name is {list(classes_dir.keys())[list(classes_dir.values()).index(class_idx)]}')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    mode = 'train'
    if mode == 'train':
        main(args)
    else:
        # 验证需要更改resume
        file_path = '/home/tonnn/.nas/weijia/datasets/project1/test/apple/*.jpg'
        images = glob.glob(file_path)
        for img in images:
            print('\n')
            main(args, mode=mode, test_img_path=img)
