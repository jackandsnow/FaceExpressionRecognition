import os
import sys
import cv2
from importlib import reload
from PIL import Image

reload(sys)

# 待检测的图片路径
image_path = r'E:\JackWorkspace\data_sets\self\3.jpg'
image_dir = r'C:\Users\lenovo\Pictures\Camera Roll'

# 获取训练好的人脸的参数数据，这里直接从GitHub上使用默认值
face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')


def detect_face(file_name=image_path):
    img = cv2.imread(file_name)
    if img is None:
        print('该图片不存在！')
        sys.exit(-1)

    im = Image.open(file_name, mode='r')
    while True:
        w, h = im.size
        if w > 800 or h > 800:
            im = im.resize((w // 2, h // 2))
            im.save(file_name)
        else:
            break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 探测图片中的人脸
    face_arr = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(5, 5)
    )
    faces = []
    for (x, y, w, h) in face_arr:
        faces.append((x, y, x + w, y + h))
    # print("发现{0}个人脸！".format(len(faces)))
    return faces, img


# 保存预测结果
def save_face(faces, image, emotion, save_name):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, emotion, (200, 100), font, 3, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(save_name, image)


def show_face(faces, image):
    if len(faces) == 0:
        print('没有人脸要显示！')
        sys.exit(-1)
    for (x, y, w, h) in faces:
        # (0, 255, 0) - green, 2 pixels
        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
        # cv2.circle(img, ((x + x + w) // 2, (y + y + h) // 2), w // 2, (0, 255, 0), 2)

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(image, 'Emotion', (250, 100), font, 3, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('FACE', image)
    cv2.waitKey(0)


def resize_image(path=image_dir, save_path=image_dir, emotion=None):
    count = 0
    for r, d, files in os.walk(path):
        for f in files:
            names = f.split('.')
            if names[len(names) - 1] == 'jpg' or names[len(names) - 1] == 'png':
                count += 1
                image_open = Image.open(r + '\\' + f, 'r')
                width, height = image_open.size
                crop = image_open.crop((width // 2 - height // 2, 0, width // 2 + height // 2, height))
                if emotion is None:
                    crop.save(save_path + '\\' + f)
                else:
                    crop.save(save_path + '\\' + emotion + str(count) + '.jpg')
        break
    print('Resize image finished!')
