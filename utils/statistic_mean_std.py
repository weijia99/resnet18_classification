# 统计均值和标准差
import glob
import os
import numpy
from PIL import Image

if __name__ == '__main__':
    train_files = glob.glob(os.path.join("./train1", "*", "*.jpg"))
    result = []
    for file in train_files:
        img = Image.open(file).convert('RGB')
        img = numpy.array(img).astype(numpy.uint8)

        # 执行灰度标准化01
        img = img/255
        result.append(img)

    # 输出的是batchsize*像素
    print(numpy.shape(result))  # [bs,h,w,c]
    # 这样就会值保留通道数，就可以进行输出了
    mean = numpy.mean(result, axis=(0, 1, 2))
    std = numpy.std(result, axis=(0, 1, 2))
    print(mean)
    print(std)
