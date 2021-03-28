import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.backends import cudnn
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import CarvanaDataset
from torch.utils.data import DataLoader, random_split

dir_img = 'data/imgs/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'


def train_net(net, device, epochs=5, batch_size=1, lr=0.001, val_percent=0.1, save_cp=True, img_scale=0.5):
    # 载入数据集
    dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # 验证集长度
    n_val = int(len(dataset) * val_percent)
    # 训练集长度
    n_train = len(dataset) - n_val
    # 将数据集按照预先设定好的长度随机划分
    train, val = random_split(dataset, [n_train, n_val])
    # 对数据进行批处理
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    # 创建tensorboard, 注释(学习率，batchsize和图像尺寸)加在文件名后面
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    # 日志信息
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    # 梯度平方的滑动平均优化器
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 学习率调整器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        # 多分类交叉熵(softmax)
        criterion = nn.CrossEntropyLoss()
    else:
        # 二分类交叉熵(Sigmoid)
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        # 设置train模式
        net.train()

        # epoch_loss初始化
        epoch_loss = 0
        # 进度条
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                # 获取image及配套的mask数据
                imgs, true_masks = batch['image'], batch['mask']
                # 断言判断, image的channel是否与网络的channel匹配
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                # 将image和配套mask数据搬运至device
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                # image经过网络输出的预测mask数据
                masks_pred = net(imgs)
                # 计算预测mask数据和配套mask数据的交叉熵
                loss = criterion(masks_pred, true_masks)
                # epoch_loss进行累加
                epoch_loss += loss.item()
                # 将loss写入tensorboard
                writer.add_scalar('Loss/train', loss.item(), global_step)

                # 设置进度条后缀
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # 训练
                # 将梯度初始化为零
                optimizer.zero_grad()
                # 进行第一次反向传播求梯度
                loss.backward()
                # 进行梯度裁剪
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                # 使用优化器反向传播求梯度
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        # 将weight和gradien写入tensorboard
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    # 获取eval模式下的平均loss
                    val_score = eval_net(net, val_loader, device)
                    # 学习率调整器优化学习率
                    scheduler.step(val_score)
                    # 将learning_rate写入tensorboard
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    # 将val_loss写入日志和tensorboard
                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    # 将image数据写入tensorboard添加到摘要
                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        # 将mask数据写入tensorboard添加到摘要
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        # 如果需要保存checkpoints文件
        if save_cp:
            try:
                # 创建checkpoints目录
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            # 将参数保存在字典
            torch.save(net.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
    # 关闭tensorboard
    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5, help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1, help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001, help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    # 对root logger进行一次性配置
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # 获取参数
    args = get_args()
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    logging.info(f'Network:\n' f'\t{net.n_channels} input channels\n' f'\t{net.n_classes} output channels (classes)\n' f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    # 如果需要载入checkpoints文件
    if args.load:
        # 将参数载入网络
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    # 将网络搬运至指定设备
    net.to(device=device)
    # 提高卷积神经网络运行速度, 但需要更多内存
    cudnn.benchmark = True

    try:
        # 训练
        train_net(net=net, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr, device=device, img_scale=args.scale, val_percent=args.val / 100)
    except KeyboardInterrupt:
        # 将参数保存在字典
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
