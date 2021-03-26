import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    # 设置eval模式
    net.eval()
    # loader的长度
    n_val = len(loader)
    tot = 0

    # 进度条
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            # 获取image及配套的mask数据
            imgs, true_masks = batch['image'], batch['mask']
            # 将image和配套mask数据搬运至device
            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            # 不需要计算梯度，也不会进行反向传播
            with torch.no_grad():
                # image经过网络输出的预测mask数据
                mask_pred = net(imgs)

            if net.n_classes > 1:
                # 计算预测mask数据和配套mask数据的交叉熵
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    # 设置train模式
    net.train()
    return tot / n_val
