import os
import cv2
import numpy as np
from PIL import Image

label_dir = r'E:\JackWorkspace\data_sets\Emotion_labels'
image_dir = r'E:\JackWorkspace\data_sets\cohn-kanade-images'
# 所有数据
all_data = []
# 所有标签
all_label = []
# 没有标签的数据
no_label_data = []
# 期望图片大小
purpose_size = 120
face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')


# 裁剪人脸部分
def image_cut(file_name):
    # cv2读取图片
    im = cv2.imread(file_name)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # cv2检测人脸
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(5, 5)
    )

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # PIL读取图片
            img = Image.open(file_name)
            # 转换为灰度图片
            img = img.convert("L")
            # 裁剪人脸部分
            crop = img.crop((x, y, x + w, y + h))
            # 缩小为120*120
            crop = crop.resize((purpose_size, purpose_size))
            return crop
    return None


# 将图片转换为数据矩阵
def image_to_matrix(filename):
    # 裁剪并缩小
    img = image_cut(filename)
    # print(img.size)
    width, height = img.size
    data = img.getdata()
    data_matrix = np.reshape(data, (1, height * width))
    # data_matrix = np.reshape(data, (height, width))
    return np.array(data_matrix, dtype=int)


# 获取文件中的label值
def get_label(file_name):
    ft = open(file_name, 'r+')
    line = ft.readline()
    line_data = line.split(' ')
    label = float(line_data[3])
    ft.close()
    return int(label)


def get_file_data():
    # 目录列表
    dir_list = []

    for r, dirs, f in os.walk(image_dir):
        for d in dirs:
            for r1, dds, f1 in os.walk(r + '\\' + d):
                for dd in dds:
                    dir_list.append(d + '\\' + dd)
                break
        break

    # 遍历目录获取文件
    for path in dir_list:
        # 处理 images
        for r, d, files in os.walk(image_dir + '\\' + path):
            # 取后半部分比较明显的表情
            files_length = len(files)
            for fi in range((files_length - 1) // 2, files_length):
                filename = r + '\\' + files[fi]
                if filename.split('.')[1] == 'png':
                    img_data = image_to_matrix(filename)
                    # 处理 labels
                    for r1, d1, ffs in os.walk(label_dir + '\\' + path):
                        if len(ffs) > 0:
                            for f in ffs:
                                label = get_label(r1 + '\\' + f)
                                all_label.append(label)
                            all_data.append(img_data)
                        else:
                            no_label_data.append(img_data)
                        break

    print(len(all_data))
    print(len(all_label))
    print(len(no_label_data))

    test_num = len(all_label) % 100

    data = np.reshape(all_data, (len(all_data), purpose_size * purpose_size))
    label = np.reshape(all_label, (len(all_label), 1))
    no_label = np.reshape(no_label_data, (len(no_label_data), purpose_size * purpose_size))
    label = label - 1  # 将1-7的值转换为0-6
    # 随机打乱存放
    total = np.column_stack((data, label))
    np.random.shuffle(total)
    data, label = total[:, :-1], total[:, -1:]
    print(data.shape)
    print(label.shape)
    print(no_label.shape)
    np.savetxt(str(purpose_size) + 'data/train_data.csv', data[:-test_num], fmt='%d', delimiter=' ')
    np.savetxt(str(purpose_size) + 'data/train_label.csv', label[:-test_num], fmt='%d', delimiter=' ')
    np.savetxt(str(purpose_size) + 'data/test_data.csv', data[-test_num:], fmt='%d', delimiter=' ')
    np.savetxt(str(purpose_size) + 'data/test_label.csv', label[-test_num:], fmt='%d', delimiter=' ')
    # np.savetxt('ccn_data/no_label_data.csv', no_label, fmt='%d', delimiter=' ')


if __name__ == '__main__':
    get_file_data()
    print('OK')
