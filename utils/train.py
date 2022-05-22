# 1.首先是main函数的构建

# 判断是不是train的模式
import math
import sys
from collections.abc import Iterable

import os
import timm

import torch.utils.data
import torchvision.transforms
from timm.utils import accuracy
from torch.utils.tensorboard import SummaryWriter
from misc import NativeScalerWithGradNormCount as NativeScaler
import misc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transform(is_train, args):
    if is_train:
        # 进行训练集的transform，设置
        print("train transform")
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(args.input_size, args.input_size),
             torchvision.transforms.RandomHorizontalFlip(),
             torchvision.transforms.RandomVerticalFlip(),
             torchvision.transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
             torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
             torchvision.transforms.ToTensor()]

        )
    else:
        print("eval mode")
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(args.input_size, args.input_size),
             torchvision.transforms.ToTensor()]

        )
    return transform


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    path = os.path.join(args.root_path, 'train1' if is_train else 'test1')
    dataset = torchvision.datasets.ImageFolder(path, transform=transform)
    info = dataset.find_classes(path)
    print(f"finding classes from {path} :\t{info[0]}")
    return dataset


@torch.no_grad
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter=" ")
    header = "test:"

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
    return {k: meter.global_avg for k, meter in metric_logger.items()}


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.optimizer,
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

        model = timm.create_model("resnet18", pretrained=True, drop_rate=0.1, drop_path_rate=0.1)

        n_parameters = sum(p.numel() for p in model.parameters() if p.require_grade())
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
