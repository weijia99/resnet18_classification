import glob
import random

from PIL import Image
import numpy as np
import os

# 1.初始化设置
split_ratio = 0.95
desired_size = 128
file_path = '/home/tonnn/.nas/weijia/datasets/project1/raw'

# 2.使用glob进行通配链接,使用os。join链接所有的目录
img_dir = glob.glob(os.path.join(file_path, '*'))
img_dir = [d for d in img_dir if os.path.isdir(d)]

catalogue_class = len(img_dir)
# 对此文件进行目录

# 3.接下来就是对图像进行缩放,这是一个类
for file in img_dir:
    # 通过对这个字符串使用split进行切割，获得最后一部分，就是类名
    file1 = file
    file = file.split('/')[-1]
    # 对这个类进行构建train&test
    os.makedirs(f'train1/{file}', exist_ok=True)
    os.makedirs(f'test1/{file}', exist_ok=True)
    # {'PNG', 'JPG', 'png', 'jpeg', 'jpg'}五种
    img_files = glob.glob(os.path.join(file_path, file, '*.jpg'))
    # img_files += glob.glob(os.path.join(file_path, file, '*.jpeg'))
    img_files += glob.glob(os.path.join(file_path, file, '*.png'))
    # img_files += glob.glob(os.path.join(file_path, file, '*.PNG'))
    img_files += glob.glob(os.path.join(file_path, file, '*.JPG'))

    random.shuffle(img_files)
    # 对数据进行shuffle
    boundary = int(len(img_files) * split_ratio)
    for i, img in enumerate(img_files):
        img1 = Image.open(img).convert('RGB')
        old_size = img1.size

        trans_ratio = float(desired_size / max(old_size))

        # 进行转换
        new_size = tuple([int(x * trans_ratio) for x in old_size])
        img1 = img1.resize(new_size, Image.ANTIALIAS)
        # 新建一个128*128的，然后进行填入
        img_new = Image.new('RGB', (desired_size, desired_size))
        img_new.paste(img1, ((desired_size - img1.size[0]) // 2,
                             (desired_size - img1.size[1]) // 2))

        # 放入图片
        if i > boundary:
            img_new.save(os.path.join(f'test1/{file}', img.split('/')[-1].split('.')[0] + '.jpg'))
        else:
            img_new.save(os.path.join(f'train1/{file}', img.split('/')[-1].split('.')[0] + '.jpg'))

    print(f'{file} process end')
test_files = glob.glob(os.path.join('test1', '*', '*.jpg'))
train_files = glob.glob(os.path.join('train1', '*', '*.jpg'))

print(f'Totally {len(test_files)} files for testing')
print(f'Totally {len(train_files)} files for training')
