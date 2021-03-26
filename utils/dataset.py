from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


# 构建BasicDataset父类
class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        # images文件路径
        self.imgs_dir = imgs_dir
        # masks文件路径
        self.masks_dir = masks_dir
        # 缩放比例
        self.scale = scale
        # masks文件后缀
        self.mask_suffix = mask_suffix
        # 断言判断, 缩放比例是否在0-1之间
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        # 创建文件名开头不为"."(隐藏文件)的文件名列表
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        # 返回数据集长度
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        # 获取image的长宽
        w, h = pil_img.size
        # 将image的原始长宽进行缩放
        newW, newH = int(scale * w), int(scale * h)
        # 断言判断, image的新长宽是否大于0
        assert newW > 0 and newH > 0, 'Scale is too small'
        # 将image进行缩放
        pil_img = pil_img.resize((newW, newH))

        # 将image转化为2维矩阵
        img_nd = np.array(pil_img)

        # 将image添加第三维channel
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # 将HWC转化为CHW
        img_trans = img_nd.transpose((2, 0, 1))
        # image进行归一化
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        # 第i个image文件名
        idx = self.ids[i]
        # 通过路径搜索mask文件
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        # 通过路径搜索image文件
        img_file = glob(self.imgs_dir + idx + '.*')

        # 断言判断, mask文件存在且只有一个
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        # 断言判断, image文件存在且只有一个
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        # 断言判断, image和mask的尺寸相同
        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        # 处理image和mask图片数据
        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img).type(torch.FloatTensor), 'mask': torch.from_numpy(mask).type(torch.FloatTensor)}


# 通过BasicDataset父类构建CarvanaDataset子类
class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
