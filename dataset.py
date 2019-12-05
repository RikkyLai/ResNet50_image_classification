import cv2
import numpy as np
from data_get import loadCSVfile
from keras.applications.resnet50 import preprocess_input, decode_predictions


class DataSet(object):
    def __init__(self, images, labels):
        self.num_samples = images.shape[0]
        self.images = images
        self.labels = labels
        # self.cls = cls
        self.epoch_done = 0
        self.index_in_epoch = 0

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size

        if self.index_in_epoch > self.num_samples:
            self.epoch_done += 1
            start = 0
            self.index_in_epoch = batch_size
        end = self.index_in_epoch
        return self.images[start:end], self.labels[start:end]


def load_train(train_path, train_label, img_size, classes):
    """
    因为系统内存无法存储那么大的图像矩阵，只能一个batch地去读取图片
    :param train_path: 经过batch_index 规定好范围的图片路径
    :param train_label: 图片label
    :param img_size: 默认224
    :param classes: 有多少个类
    :return: [batch, img_size, img_size, channel]的图像， [batch, num_classes]的label（经过one_hot）
    """
    images = []
    labels = []
    for i in range(len(train_path)):
        image = cv2.imread(train_path[i])
        image = cv2.resize(image, (img_size, img_size), 0.0, interpolation=cv2.INTER_CUBIC)
        image = image.astype(np.float32)
        image = preprocess_input(image)
        images.append(image)

        label = np.zeros(classes)
        label[train_label[i]] = 1.0
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def load_validation(val_path, img_size=224):
    """
    这个函数首先是将验证集的路径先打乱，因为系统的内存无法记录3万多张图片的矩阵，所以想的办法是先打乱，然后取
    前300张来做验证
    :param val_path: 验证集图片的路径
    :param classes:  验证集label
    :param img_size: 默认是224
    :return: [batch, img_size, img_size, channel]的图像， [batch, num_classes]的label（经过one_hot）
    """
    # val_data_path, val_label = loadCSVfile(val_path)
    images = []
    labels = []
    # 上面两个就是一个batch得到的[batch, img_size, img_size, channel]的图像， [batch, num_classes]的label（经过one_hot）
    # 路径打乱
    # index = [i for i in range(val_data_path.shape[0])]
    # np.random.shuffle(index)
    # val_data_path = val_data_path[index]
    # val_label = val_label[index]

    # for i in range(300):
    image = cv2.imread(val_path)
    image = cv2.resize(image, (img_size, img_size), 0.0, interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = preprocess_input(image)
    images.append(image)

    # label = np.zeros(classes)
    # label[val_label[i]] = 1.0
    # labels.append(label)

    images = np.array(images)
    # labels = np.array(labels)

    return images


def read_train_sets(train_path, train_label, img_size, classes):
    class DataSets(object):
        pass

    data_sets = DataSets()
    images, labels = load_train(train_path, train_label, img_size, classes)
    # index = [i for i in range(images.shape[0])]
    # np.random.shuffle(index)
    # train_images = images[index]
    # train_labels = labels[index]

    # data_sets.train = DataSet(train_images, train_labels)

    return images, labels

