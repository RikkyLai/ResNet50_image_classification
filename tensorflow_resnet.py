import tensorflow as tf
import numpy as np
from data_get import loadCSVfile
from dataset import load_train, load_validation
import cv2
from keras.applications.resnet50 import preprocess_input
import os
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_path = "train_csv.csv"
validation_path = "val_csv.csv"
test_path = "test_csv.csv"
batch_size = 8
img_size = 224
channel = 3
num_classes = 51
lr = 0.001
epoch = 5


def conv_op(x, name, n_out, training, useBN, kh=3, kw=3, dh=1, dw=1, padding="SAME", activation=tf.nn.relu,
            ):
    '''
    x:输入
    kh,kw:卷集核的大小
    n_out:输出的通道数
    dh,dw:strides大小
    name:op的名字

    '''
    n_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        w = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b = tf.get_variable(scope + "b", shape=[n_out], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.01))
        conv = tf.nn.conv2d(x, w, [1, dh, dw, 1], padding=padding)
        z = tf.nn.bias_add(conv, b)
        if useBN:
            z = tf.layers.batch_normalization(z, trainable=training)
        if activation:
            z = activation(z)
        return z


def max_pool_op(x, name, kh=2, kw=2, dh=2, dw=2, padding="SAME"):
    return tf.nn.max_pool(x,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding=padding,
                          name=name)


def avg_pool_op(x, name, kh=2, kw=2, dh=2, dw=2, padding="SAME"):
    return tf.nn.avg_pool(x,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding=padding,
                          name=name)


def fc_op(x, name, n_out, activation=tf.nn.relu):
    n_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        w = tf.get_variable(scope + "w", shape=[n_in, n_out],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(scope + "b", shape=[n_out], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.01))

        fc = tf.matmul(x, w) + b

        out = activation(fc)

    return fc, out


def res_block_layers(x, name, n_out_list, change_dimension=False, block_stride=1):
    if change_dimension:
        short_cut_conv = conv_op(x, name + "_ShortcutConv", n_out_list[1], training=True, useBN=True, kh=1, kw=1,
                                 dh=block_stride, dw=block_stride,
                                 padding="SAME", activation=None)
    else:
        short_cut_conv = x

    block_conv_1 = conv_op(x, name + "_lovalConv1", n_out_list[0], training=True, useBN=True, kh=1, kw=1,
                           dh=block_stride, dw=block_stride,
                           padding="SAME", activation=tf.nn.relu)

    block_conv_2 = conv_op(block_conv_1, name + "_lovalConv2", n_out_list[0], training=True, useBN=True, kh=3, kw=3,
                           dh=1, dw=1,
                           padding="SAME", activation=tf.nn.relu)

    block_conv_3 = conv_op(block_conv_2, name + "_lovalConv3", n_out_list[1], training=True, useBN=True, kh=1, kw=1,
                           dh=1, dw=1,
                           padding="SAME", activation=None)

    block_res = tf.add(short_cut_conv, block_conv_3)
    res = tf.nn.relu(block_res)
    return res


def bulid_resNet(x, num_class, training=True, usBN=True):
    conv1 = conv_op(x, "conv1", 64, training, usBN, 3, 3, 1, 1)
    pool1 = max_pool_op(conv1, "pool1", kh=3, kw=3)

    block1_1 = res_block_layers(pool1, "block1_1", [64, 256], True, 1)
    block1_2 = res_block_layers(block1_1, "block1_2", [64, 256], False, 1)
    block1_3 = res_block_layers(block1_2, "block1_3", [64, 256], False, 1)

    block2_1 = res_block_layers(block1_3, "block2_1", [128, 512], True, 2)
    block2_2 = res_block_layers(block2_1, "block2_2", [128, 512], False, 1)
    block2_3 = res_block_layers(block2_2, "block2_3", [128, 512], False, 1)
    block2_4 = res_block_layers(block2_3, "block2_4", [128, 512], False, 1)

    block3_1 = res_block_layers(block2_4, "block3_1", [256, 1024], True, 2)
    block3_2 = res_block_layers(block3_1, "block3_2", [256, 1024], False, 1)
    block3_3 = res_block_layers(block3_2, "block3_3", [256, 1024], False, 1)
    block3_4 = res_block_layers(block3_3, "block3_4", [256, 1024], False, 1)
    block3_5 = res_block_layers(block3_4, "block3_5", [256, 1024], False, 1)
    block3_6 = res_block_layers(block3_5, "block3_6", [256, 1024], False, 1)

    block4_1 = res_block_layers(block3_6, "block4_1", [512, 2048], True, 2)
    block4_2 = res_block_layers(block4_1, "block4_2", [512, 2048], False, 1)
    block4_3 = res_block_layers(block4_2, "block4_3", [512, 2048], False, 1)

    pool2 = avg_pool_op(block4_3, "pool2", kh=7, kw=7, dh=1, dw=1, padding="SAME")
    shape = pool2.get_shape()
    fc_in = tf.reshape(pool2, [-1, shape[1].value * shape[2].value * shape[3].value])
    logits, prob = fc_op(fc_in, "fc1", num_class, activation=tf.nn.softmax)
    # 需要进入损失函数的是没有经过激活函数的logits
    return logits, prob


def training_pro():
    train_data_path, train_label = loadCSVfile(train_path)
    batch_index = []
    # 将 训练数据 分batch
    for i in range(train_data_path.shape[0]):
        if i % batch_size == 0:
            batch_index.append(i)
    if batch_index[-1] is not train_data_path.shape[0]:
        batch_index.append(train_data_path.shape[0])

    input = tf.placeholder(dtype=tf.float32, shape=[None, img_size, img_size, channel], name="input")
    # output = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name="output")
    output = tf.placeholder(dtype=tf.int64, shape=[None], name="output")
    # 将label值进行onehot编码
    one_hot_labels = tf.one_hot(indices=tf.cast(output, tf.int32), depth=51)

    # 需要传入到softmax_cross_entropy_with_logits的是没有经过激活函数的y_pred
    y_pred, _ = bulid_resNet(input, num_classes)
    y_pred = tf.reshape(y_pred, shape=[-1, num_classes])

    tf.add_to_collection('output_layer', y_pred)

    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=output))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=one_hot_labels))
    # 该api,做了三件事儿 1. y_ -> softmax 2. y -> one_hot 3. loss = ylogy

    tf.summary.scalar('loss', loss)

    # 这一段是为了得到accuracy，首先是得到数值最大的索引
    # 准确度
    a = tf.argmax(y_pred, 1)
    b = tf.argmax(one_hot_labels, 1)
    correct_pred = tf.equal(a, b)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # 将上句的布尔类型 转化为 浮点类型,然后进行求平均值,实际上就是求出了准确率

    # 标记一下：这里可以尝试一下GD方法，体验一下学习率调参，然后加个momentum功能试一下
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(max_to_keep=10)

    total_loss = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 从上次中断处继续训练
        # saver = tf.train.import_meta_graph('./model_lstm/stock2.model-110.meta')
        # # MODEL_SAVE_PATH = './model_lstm/'
        # # ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)  # 获取checkpoints对象
        # # if ckpt and ckpt.model_checkpoint_path:  # 判断ckpt是否为空，若不为空，才进行模型的加载，否则从头开始训练
        # saver.restore(sess, './model_lstm/stock2.model-20')  # 恢复保存的神经网络结构，实现断点续训

        merged = tf.summary.merge_all()
        train_writer_train = tf.summary.FileWriter("logs_res/train", sess.graph)
        train_writer_val = tf.summary.FileWriter("logs_res/val")

        i = 0
        while True:
            for step in range(len(batch_index) - 1):
                i += 1
                x_train, _ = load_train(train_data_path[batch_index[step]:batch_index[step + 1]],
                                               train_label[batch_index[step]:batch_index[step + 1]], img_size, num_classes)
                y_train = np.array(list(train_label[batch_index[step]:batch_index[step + 1]]))
                _a, _, loss_, y_t, y_p, a_, b_ = sess.run(
                    [merged, train_op, loss, one_hot_labels, y_pred, a, b],
                    feed_dict={input: x_train, output: y_train})

                print('step: {}, train_loss: {}'.format(i, loss_))
                if i % 20 == 0:
                    _loss, acc_train = sess.run([loss, accuracy], feed_dict={input: x_train, output: y_train})
                    print('--------------------------------------------------------')
                    print('step: {}  train_acc: {}  loss: {}'.format(i, acc_train, _loss))
                    print('--------------------------------------------------------')
                    if (i+1) % 10000 == 0:
                        saver.save(sess, 'model_1202/resnet50.model', global_step=i)

        # for i in range(epoch):
        #     for step in range(len(batch_index)-1):
        #         # 通过batch_index记录每个batch的索引范围，再通过load_train得到一个batch的图像矩阵以及one_hot好的label，
        #         # 这里没有用到one_hot好的label，是因为tf.losses.sparse_softmax_cross_entropy这个api能够完成one_hot,
        #         # 所以就没有用到自己的ont_hot,这样做是方便完成对accuracy的计算
        #         x_train, _ = load_train(train_data_path[batch_index[step]:batch_index[step + 1]],
        #                                        train_label[batch_index[step]:batch_index[step + 1]], img_size, num_classes)
        #         y_train = np.array(list(train_label[batch_index[step]:batch_index[step + 1]]))
        #         result_train, acc_val, _, loss_value = sess.run([merged, accuracy, train_op, loss], feed_dict={input: x_train, output: y_train})
        #         train_writer_train.add_summary(result_train, i*(len(batch_index)-1)+step)
        #         total_loss += loss_value
        #         print("step: ", i*(len(batch_index)-1)+step, "training cost: ", loss_value)
        #         print("At step: ", i * (len(batch_index) - 1) + step, "accuracy: ", acc_val)
        #         if step % 99 == 0 and step is not 0:
        #
        #             # x_val, y_val = load_validation(validation_path, 51)
        #             # result_val, loss_value_val = sess.run([merged, loss], feed_dict={input: x_val, output: y_val})
        #             # train_writer_val.add_summary(result_val, i*(len(batch_index)-1)+step)
        #             print("At step: ", i * (len(batch_index) - 1) + step, "we got an average cost: ", total_loss/100)
        #             total_loss = 0
        #             print("保存模型：", saver.save(sess, 'model_1129/resnet50.model', global_step=i * (len(batch_index) - 1) + step))


def prediction():
    # image = cv2.imread(test_data_path)
    # image = cv2.resize(image, (img_size, img_size), 0.0, interpolation=cv2.INTER_LINEAR)
    # image = image.astype(np.float32)
    # image = preprocess_input(image)
    # image = np.expand_dims(image, axis=0)
    val_data_path, val_label = loadCSVfile(validation_path)
    predicted = []
    with tf.Session() as sess:
        # 参数恢复
        saver = tf.train.import_meta_graph('model_1202/resnet50.model-200000.meta')
        saver.restore(sess, 'model_1202/resnet50.model-200000')

        graph = tf.get_default_graph()
        # 从tensor的checkpoint查看name
        x_in = graph.get_tensor_by_name(name='input:0')
        # y_ = graph.get_tensor_by_name(name='output_layer:0')
        y_ = tf.get_collection("output_layer")[0]
        for i in range(len(val_data_path)):

            result = sess.run(y_, feed_dict={x_in: load_validation(val_data_path[i])})

            Y_predict = np.argmax(result)
            predicted.append(Y_predict)
    predicted = np.array(predicted)
    np.save('prediction.npy', predicted)
    correct_prediction = np.equal(predicted, val_label)
    # 将布尔值转化为int类型,也就是 0 或者 1, tf.equal() 返回值是布尔类型
    accuracy = np.mean(correct_prediction.astype(int))
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    # training_pro()
    # prediction()

    val_data_path, val_label = loadCSVfile(validation_path)
    predicted = np.load('prediction.npy')
    data1 = {'val_data_path': val_data_path, 'val_data_label': val_label, 'val_prediction': predicted}
    df = pd.DataFrame(data=data1)
    df.to_csv('data1.csv', header=1)
