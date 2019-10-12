# coding=utf-8
import os
import glob
import cv2


def main():
    data_dir = os.path.join('/usr', 'local', 'wk', 'data', 'fortnite')
    org_train_dir = os.path.join(data_dir, 'train')
    org_val_dir = os.path.join(data_dir, 'val')
    resize_size = (800, 450)  # width, height

    resized_train_dir = os.path.join(data_dir, '_'.join(['train', str(resize_size[0]), str(resize_size[1])]))
    resized_val_dir = os.path.join(data_dir, '_'.join(['val', str(resize_size[0]), str(resize_size[1])]))
    print(resized_train_dir, resized_val_dir)

    train_dir_list = os.listdir(org_train_dir)
    train_dir_list = [train for train in train_dir_list if train.find('DS_Store') == -1]

    val_dir_list = os.listdir(org_val_dir)
    val_dir_list = [val for val in val_dir_list if val.find('DS_Store') == -1]

    for i, train_dir in enumerate(train_dir_list):
        print(i, len(train_dir_list))
        dir_images = glob.glob(os.path.join(org_train_dir, train_dir, '*.jpg'))
        for image_file in dir_images:
            img = cv2.imread(image_file)
            resized_image = cv2.resize(img, (resize_size[0], resize_size[1]))
            resized_image_file = os.path.join(resized_train_dir, train_dir, os.path.basename(image_file))
            if not os.path.exists(os.path.join(resized_train_dir, train_dir)):
                os.makedirs(os.path.join(resized_train_dir, train_dir), exist_ok=True)
            cv2.imwrite(resized_image_file, resized_image)

    """
    for i, val_dir in enumerate(val_dir_list):
        print(i, len(val_dir_list))
        dir_images = glob.glob(os.path.join(org_val_dir, val_dir, '*.jpg'))
        for image_file in dir_images:
            img = cv2.imread(image_file)
            resized_image = cv2.resize(img, (resize_size[0], resize_size[1]))
            resized_image_file = os.path.join(resized_val_dir, val_dir, os.path.basename(image_file))
            if not os.path.exists(os.path.join(resized_val_dir, val_dir)):
                os.makedirs(os.path.join(resized_val_dir, val_dir), exist_ok=True)
            cv2.imwrite(resized_image_file, resized_image)
    """

if __name__ == '__main__':
    main()
