
### 深度神经网络Keras实现手写数字识别

在学习机器学习的时候，首当其冲的就是准备一份通用的数据集，方便与其他的算法进行比较。

### MNIST简介
MNIST数据集原网址：http://yann.lecun.com/exdb/mnist/
![](assets/markdown-img-paste-20190216133336752.png)

数据集是这样的一些手写数字

**问题：通过某个算法将0-9的数字进行分类**

### 下载
Github源码下载：数据集（源文件+解压文件+字体图像jpg格式），py源码文件
文件目录
```python
/utils/data_util.py 用于加载MNIST数据集方法文件
/utils/test.py 用于测试的文件，一个简单的KNN测试MNIST数据集
/data/train-images.idx3-ubyte 训练集X
/dataset/train-labels.idx1-ubyte 训练集y
/dataset/data/t10k-images.idx3-ubyte 测试集X
/dataset/data/t10k-labels.idx1-ubyte 测试集y
```

### 结构解释
MNIST数据集解释
将MNIST文件解压后，发现这些文件并不是标准的图像格式。这些图像数据都保存在二进制文件中。每个样本图像的宽高为28*28。

mnist的结构如下，选取train-images
```python
[code]TRAINING SET IMAGE FILE (train-images-idx3-ubyte):

[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
```

首先该数据是以二进制存储的，我们读取的时候要以’rb’方式读取；其次，真正的数据只有[value]这一项，其他的[type]等只是来描述的，并不真正在数据文件里面。也就是说，在读取真实数据之前，我们要读取4个

32 bit integer
.由[offset]我们可以看出真正的pixel是从0016开始的，一个int 32位，所以在读取pixel之前我们要读取4个 32 bit integer，也就是magic number, number of images, number of rows, number of columns. 当然，在这里使用struct.unpack_from()会比较方便.

### 算法实现
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def load_data():
    #  通过keras自带的mnist工具包加载数据
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    showimage(x_train[0])
    x_train = x_train.reshape(60000, 784).astype('float32')
    x_test = x_test.reshape(10000, 784).astype('float32')
    #  数据归一化 Min-Max归一化
    x_train /= 255
    x_test /= 255
    # 转换成one-hot编码
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


# 可视化手写数字图片
def showimage(imagedata):
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(111)
    ax1.imshow(imagedata, cmap='gray')
    width, height = imagedata.shape
    thresh = imagedata.max() / 2.5
    for x in range(width):
        for y in range(height):
            ax1.annotate(str(round(imagedata[x][y], 2)), xy=(y, x), horizontalalignment='center',
                         verticalalignment='center',
                         color='white' if imagedata[x][y] < thresh else 'black')
    plt.show()


(x_train, y_train), (x_test, y_test) = load_data()
# 初始化参数
batch_size = 128
num_classes = 10
epochs = 20

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=batch_size,
                    epochs=epochs, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
