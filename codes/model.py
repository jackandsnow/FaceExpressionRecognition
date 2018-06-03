import os
import numpy as np
import tensorflow as tf
from PIL import Image

from codes.face import detect_face, save_face

data_dir = 'data/ck+data.csv'  # 图像目录
label_dir = 'data/ck+label.csv'  # 标签目录
image_size = 120  # 方形图像的像素宽度
channel = 1  # 单通道
emotion_type = 7  # 表情类型总数
batch_size = 30  # 每次训练的样本数
epochs = 150  # 样本训练次数
learning_rate = 1e-3  # 学习率
original_features = 32
output_features = 1024

# 3000
train_num = 2700
validation_num = 300

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

Emotions = {1: 'anger', 2: 'contempt', 3: 'disgust', 4: 'fear', 5: 'happiness', 6: 'sadness', 7: 'surprise'}


# 分块读取文件内容
def read_file(filename=data_dir, block_size=200):
    with open(filename, 'r') as f:
        while True:
            data = f.readlines(block_size)
            if data:
                yield data
            else:
                return


# 初始化数据及标签
def init_data(d_dir=data_dir, l_dir=label_dir, block_size=200):
    data = []
    labels = []
    file_data = read_file(filename=d_dir, block_size=block_size)
    file_label = read_file(filename=l_dir, block_size=block_size)

    for f_data in file_data:
        splits = list(f_data)[0].split(' ')
        data.append(list(map(float, splits)))
    for f_label in file_label:
        for i in range(len(f_label)):
            labels.append(int(f_label[i]))

    # 标准化，0-1
    data = standardize(data)
    # one-hot
    labels = one_hot(labels)
    return data, labels


# 权重初始化
def weight_variable(shape, name=''):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)


# 偏置值初始化
def bias_variable(shape, name=''):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


# 卷积
def conv_2d(x, w, b, s=1, name=''):
    fx = tf.nn.conv2d(x, w, strides=[1, s, s, 1], padding='SAME')
    fy = tf.nn.bias_add(fx, b)
    return tf.nn.relu(fy, name=name)


# 池化
def max_pool_2d(x, k=3, s=2, name=''):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME', name=name)


# 归一化处理
def standardize(data):
    data_arr = np.array(data)
    data_arr /= 255.0
    return data_arr


# one-hot
def one_hot(labels):
    label_arr = np.zeros((len(labels), emotion_type))
    for i in range(len(labels)):
        label_arr[i, labels[i]] = 1
    return label_arr


# 随机打乱数据
def shuffle(data_arr, label_arr):
    _, h = label_arr.shape
    all_arr = np.column_stack((data_arr, label_arr))
    np.random.shuffle(all_arr)
    data_arr = all_arr[:, :-h]
    label_arr = all_arr[:, -h:]
    return data_arr, label_arr


# 按照batch_size取数据
def get_batches(data, labels, b_size=batch_size):
    # 整除取整
    num = len(data) // b_size
    data, labels = data[:num * b_size], labels[:num * b_size]
    for i in range(0, len(data), b_size):
        yield data[i:i + b_size], labels[i:i + b_size]


# 保存模型
def save_model(session, path):
    saver = tf.train.Saver(save_relative_paths=True)
    save_path = saver.save(session, path)
    print('model saved in path: ', save_path)


# 划分数据集
def divide_datasets(tr_num, val_num):
    # 读取数据
    data_arr, label_arr = init_data(data_dir, label_dir, 200)
    data_arr, label_arr = shuffle(data_arr, label_arr)
    train_x = data_arr[0:tr_num]
    train_y = label_arr[0:tr_num]
    validation_x = data_arr[tr_num:tr_num + val_num]
    validation_y = label_arr[tr_num:tr_num + val_num]
    return train_x, train_y, validation_x, validation_y


def train_model():
    # 通过迭代次数修改learn_rate
    global learning_rate
    layer = 1
    # Inputs
    with tf.name_scope('INPUTS'):
        x = tf.placeholder(tf.float32, [None, pow(image_size, 2)], name='x_input')
        y = tf.placeholder(tf.float32, [None, emotion_type], name='y_input')

    # 原始图片大小，120*120, 单通道
    x_image = tf.reshape(x, [-1, image_size, image_size, channel], name='x_image')

    """
    # Layer 1
    # 卷积核(filter)的尺寸是5*5, 通道数为1，输出通道为32，即feature map 数目为32
    # 又因为strides=[1,1,1,1] 所以单个通道的输出尺寸应该跟输入图像一样。即总的卷积输出应该为?*120*120*32
    # 也就是单个通道输出为120*120，共有32个通道,共有?个批次
    # 在池化阶段，ksize=[1,3,3,1] 那么卷积结果经过池化以后的结果，其尺寸应该是?*60*60*32
    """
    with tf.name_scope('CONV_LAYER1'):
        w_conv1 = weight_variable([5, 5, channel, original_features * 1], name='w_conv1')
        b_conv1 = bias_variable([original_features * 1], name='b_conv1')
        h_conv1 = conv_2d(x_image, w_conv1, b_conv1, s=1, name='h_conv1')
        layer *= 2
    with tf.name_scope('POOL_LAYER1'):
        h_pool1 = max_pool_2d(h_conv1, k=3, s=2, name='h_pool1')

    """
    # Layer 2
    # 卷积核5*5，输入通道为32，输出通道为64。
    # 卷积前图像的尺寸为 ?*60*60*32， 卷积后为?*60*60*64
    # 池化后，输出的图像尺寸为?*30*30*64
    """
    with tf.name_scope('CONV_LAYER2'):
        w_conv2 = weight_variable([5, 5, original_features * 1, original_features * 2], name='w_conv2')
        b_conv2 = bias_variable([original_features * 2], name='b_conv2')
        h_conv2 = conv_2d(h_pool1, w_conv2, b_conv2, s=1, name='h_conv2')
        layer *= 2
    with tf.name_scope('POOL_LAYER2'):
        h_pool2 = max_pool_2d(h_conv2, k=3, s=2, name='h_pool2')

    """
    # Layer 3
    # 卷积核5*5，输入通道为64，输出通道为128。
    # 卷积前图像的尺寸为 ?*30*30*64， 卷积后为?*30*30*128
    # 池化后，输出的图像尺寸为?*15*15*128
    """
    with tf.name_scope('CONV_LAYER3'):
        w_conv3 = weight_variable([5, 5, original_features * 2, original_features * 4], name='w_conv3')
        b_conv3 = bias_variable([original_features * 4], name='b_conv3')
        h_conv3 = conv_2d(h_pool2, w_conv3, b_conv3, s=1, name='h_conv3')
        layer *= 2
    with tf.name_scope('POOL_LAYER3'):
        h_pool3 = max_pool_2d(h_conv3, k=3, s=2, name='h_pool3')

    """
    # Layer 4
    # 全连接层，输入维数?*6*6*128, 输出维数为?*1024
    """
    with tf.name_scope('FC_LAYER1'):
        conv_size = image_size // layer
        w_fc1 = weight_variable([conv_size * conv_size * original_features * 4, output_features], name='w_fc1')
        b_fc1 = bias_variable([output_features], name='b_fc1')
        h_pool3_flat = tf.reshape(h_pool3, [-1, conv_size * conv_size * original_features * 4], name='h_pool3_flat')
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1, name='h_fc1')
        # Dropout，减少过拟合
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

    """
    # Layer 5
    # 输出层，输入维数?*1024，输出维数?*7，也就是具体的0~6分类
    """
    with tf.name_scope('OUTPUT_LAYER'):
        w_fc2 = weight_variable([output_features, emotion_type], name='w_fc2')
        b_fc2 = bias_variable([emotion_type], name='b_fc2')
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2, name='y_conv')

    with tf.name_scope('LOSS'):
        # 交叉熵
        cross_entropy = -tf.reduce_sum(y * tf.log(y_conv + 1e-10), name='cross_entropy')
        tf.summary.scalar('Loss', cross_entropy)

    with tf.name_scope('TRAIN_OP'):
        # Adam优化器
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # 评估模型
    with tf.name_scope('EVAL_MODEL'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1), name='correct_prediction')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

    # terminal run: tensorboard --logdir=tensorboard
    train_writer = tf.summary.FileWriter("tensorboard/model4/train")
    val_writer = tf.summary.FileWriter("tensorboard/model4/val")
    train_writer.add_graph(sess.graph)
    tf.summary.histogram("weights1", w_conv1)
    tf.summary.histogram("weights2", w_conv2)
    tf.summary.histogram("weights3", w_conv3)
    write_op = tf.summary.merge_all()
    # 初始化所有变量
    sess.run(tf.global_variables_initializer())
    # 划分数据集
    train_x, train_y, validation_x, validation_y = divide_datasets(train_num, validation_num)
    print('train shape, x: ', train_x.shape, '  y: ', train_y.shape)
    print('validation shape, x: ', validation_x.shape, '  y: ', validation_y.shape)

    iteration = 1
    train_loss = []
    train_acc = []
    validation_loss = []
    validation_acc = []

    print('\n--------------------------Begin training---------------------------\n')
    for epo in range(epochs):
        for batch_x, batch_y in get_batches(train_x, train_y):
            summary, _, loss, acc = sess.run([write_op, train_op, cross_entropy, accuracy],
                                             feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            train_loss.append(loss)
            train_acc.append(acc)

            if iteration % 10 == 0:
                train_writer.add_summary(summary, iteration)
                train_writer.flush()
                print("Epoch: {}/{}".format(epo + 1, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Train loss: {:6f}".format(loss),
                      "Train acc: {:.6f}".format(acc))

            if iteration % 20 == 0:
                val_acc_ = []
                val_loss_ = []
                for v_x, v_y in get_batches(validation_x, validation_y):
                    summary, loss_v, acc_v = sess.run([write_op, cross_entropy, accuracy],
                                                      feed_dict={x: v_x, y: v_y, keep_prob: 1.0})
                    val_acc_.append(acc_v)
                    val_loss_.append(loss_v)
                    val_writer.add_summary(summary, iteration)
                    val_writer.flush()

                print("Epoch: {}/{}".format(epo + 1, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Validation loss: {:.6f}".format(np.mean(val_loss_)),
                      "Validation acc: {:.6f}".format(np.mean(val_acc_)))
                validation_acc.append(np.mean(val_acc_))
                validation_loss.append(np.mean(val_loss_))

            iteration += 1
        # 训练过程中降低学习率，避免梯度降为0
        if epo == 20 or epo % 50 == 0:
            learning_rate = learning_rate / 10

    print('max train accuracy is ', max(train_acc), ', avg is {:.6f}'.format(np.mean(train_acc)))
    print('max validation accuracy is ', max(validation_acc), ', avg is {:.6f}'.format(np.mean(validation_acc)))
    # 保存模型
    save_model(session=sess, path='model4/cnn' + str(epochs) + '-{:.2f}%'.format(np.mean(validation_acc) * 100))
    train_writer.close()
    val_writer.close()
    sess.close()
    print('\n--------------------------End training---------------------------\n')


def test_model(test_data_dir, test_label_dir, model_dir, model_meta, batch_length=1):
    session = tf.Session()
    saver = tf.train.import_meta_graph(model_dir + model_meta)
    saver.restore(session, tf.train.latest_checkpoint(model_dir))
    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("INPUTS/x_input:0")
    y = graph.get_tensor_by_name("INPUTS/y_input:0")
    accuracy = graph.get_tensor_by_name("EVAL_MODEL/accuracy:0")
    keep_prob = graph.get_tensor_by_name("FC_LAYER1/keep_prob:0")
    y_conv = graph.get_tensor_by_name("OUTPUT_LAYER/y_conv:0")

    y_truth = tf.argmax(y, 1)
    y_predict = tf.argmax(y_conv, 1)
    # 读取数据
    test_data, test_labels = init_data(test_data_dir, test_label_dir, 50)
    # 随机打乱所有数据
    test_data, test_labels = shuffle(test_data, test_labels)

    print('\n--------------------------Begin testing---------------------------\n')
    epoch = 1
    acc_arr = []
    for batch_x, batch_y in get_batches(test_data, test_labels, b_size=batch_length):
        y_t, y_p, acc = session.run([y_truth, y_predict, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        keys = Emotions.keys()
        acc_arr.append(acc)
        print('epoch %d: ' % epoch)
        for i in range(len(y_t)):
            for key in keys:
                if y_t[i] == key - 1:
                    print('Truth is ', Emotions.get(key), end='. ')
                if y_p[i] == key - 1:
                    print('Predict is ', Emotions.get(key), end='. ')
            print('')
        print('Average accuracy is {:.2f}%\n'.format(acc * 100))
        epoch += 1
    print('Total average accuracy is {:.2f}%'.format(np.mean(acc_arr) * 100))
    session.close()
    print('\n--------------------------End testing---------------------------\n')


def predict(file_dir, file_name, model_dir, model_meta):
    session = tf.Session()
    saver = tf.train.import_meta_graph(model_dir + model_meta)
    saver.restore(session, tf.train.latest_checkpoint(model_dir))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("INPUTS/x_input:0")
    y_conv = graph.get_tensor_by_name("OUTPUT_LAYER/y_conv:0")
    keep_prob = graph.get_tensor_by_name("FC_LAYER1/keep_prob:0")
    y_predict = tf.argmax(y_conv, 1)

    print('\n--------------------------Begin predicting---------------------------\n')
    # predict single file
    if file_name is not None:
        do_predict(session, x, y_predict, keep_prob, file_name)

    # predict multi files
    count = 0
    if file_dir is not None:
        for r, d, files in os.walk(file_dir):
            for f in files:
                names = f.split('.')
                if names[len(names) - 1] == 'jpg' or names[len(names) - 1] == 'png':
                    count += 1
                    do_predict(session, x, y_predict, keep_prob, r + '\\' + f, count)
            # break
    # close session
    session.close()
    print('\n---------------------------End predicting----------------------------\n')


def do_predict(session, x_input, y_predict, keep_prob, filename, count=0):
    save_dir = r'E:\JackWorkspace\workspace\predict_results'
    # 首先进行人脸检测
    detects, cv2_image = detect_face(file_name=filename)
    if len(detects) > 0:
        print(filename + '  检测到人脸！', end=' ')
        for (x, y, w, h) in detects:
            img = Image.open(filename, mode='r')
            # 转为灰度
            img = img.convert("L")
            # 裁剪面部区域
            crop = img.crop((x, y, w, h))
            # 重新设定大小
            crop = crop.resize((image_size, image_size))
            img_data = crop.getdata()
            img_data = np.reshape(img_data, (1, image_size * image_size))
            data = img_data / 255.0
            y_ = session.run(y_predict, feed_dict={x_input: data, keep_prob: 1.0})
            print('识别结果为', end=' ')
            emotion = Emotions.get(y_[0] + 1)
            print(emotion)
            save_face(detects, cv2_image, emotion, save_dir + '\\' + emotion + str(count) + '.jpg')
    else:
        print(filename + '  没有检测到人脸！')
