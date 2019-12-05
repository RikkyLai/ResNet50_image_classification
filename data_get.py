import numpy as np
import cv2
import pandas as pd
import os


training_data_path = 'train/'
val_data_path = 'validation/'
label_file = 'label.csv'
file_path = ['batch1/', 'batch2/', 'batch3/', 'batch4/', 'batch5/', 'batch6/', 'batch7/', 'batch8/', 'batch9/']
test_data_path = 'test/'


def loadCSVfile(training_data_path):
    tmp = pd.DataFrame(pd.read_csv(training_data_path))
    data_path = tmp.iloc[:, 0]
    label = tmp.iloc[:, 1]
    return np.array(data_path), np.array(label)


if __name__ == '__main__':
    # train_data_path, train_label = [], []
    # for file in file_path:
    #     data_path, label = loadCSVfile(test_data_path, file, label_file)
    #     train_data_path.extend(data_path.tolist())
    #     train_label.extend(label.tolist())
    # Dataset = list(zip(train_data_path, train_label))
    # # df = pd.DataFrame(data=Dataset, columns=['val_data_path', 'val_label'])
    # df = pd.DataFrame(data=Dataset, columns=['test_data_path', 'test_label'])
    # df.to_csv('test_csv.csv', index=False, header=True)

    train_data_path, train_label = loadCSVfile('train_csv.csv')
    print(train_data_path[:5])
    print(train_label[0])
    image = cv2.imread(train_data_path[0])
    cv2.imshow('test', image)
    cv2.waitKey(0)



