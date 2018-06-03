import os

from PIL import Image
from codes.face import detect_face, show_face, resize_image
from codes.model import predict, test_model, train_model


def cut_face(file_root, save_root):
    for r, d, files in os.walk(file_root):
        for f in files:
            detects, _ = detect_face(file_name=r + '\\' + f)
            if len(detects) > 0:
                print('检测到%d张人脸！' % len(detects))
                for (x, y, w, h) in detects:
                    img = Image.open(r + '\\' + f)
                    crop = img.crop((x, y, w, h))
                    crop.save(save_root + '\\' + f)
    print('Cut face finished!')


if __name__ == '__main__':
    test_data_dir = 'data/test_data.csv'
    test_label_dir = 'data/test_label.csv'
    model_dir = 'model/'
    model_meta = 'cnn200-98.29%.meta'
    # test_model(test_data_dir, test_label_dir, model_dir, model_meta)

    file_name = r'E:\JackWorkspace\workspace\test_image\surprise\surprise.jpg'
    # predict(None, file_name, model_dir, model_meta)

    # camera_dir = r'C:\Users\lenovo\Pictures\Camera Roll'
    # save_dir = r'E:\JackWorkspace\workspace\camera'
    # resize_image(camera_dir, save_dir, None)
    # predict(save_dir, None, model_dir, model_meta)

    test_image_dir = r'E:\JackWorkspace\workspace\test_image'
    predict(test_image_dir, None, model_dir, model_meta)

    print('\n--------------------------Program Finished---------------------------\n')
