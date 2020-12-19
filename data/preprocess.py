import os
import random
import zipfile


def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        raise NameError('This is not zip')


def _check(root_dir):
    data_root = os.path.join(root_dir, "data")
    if not os.path.exists(os.path.join(root_dir, "images_background.zip")) or \
            not os.path.exists(os.path.join(root_dir, "images_evaluation.zip")):
        raise FileNotFoundError
    elif not os.path.exists(data_root):
        os.mkdir(os.path.join(data_root))
        unzip_file(os.path.join(root_dir, "images_background.zip"), data_root)
        unzip_file(os.path.join(root_dir, "images_evaluation.zip"), data_root)


def split_train_test(root_dir, train_size=1200):
    _check(root_dir)
    class_folder = [os.path.join(root_dir, "data", alphabet, character)
                    for alphabet in os.listdir(os.path.join(root_dir, "data"))
                    for character in os.listdir(os.path.join(root_dir, "data", alphabet))]
    random.seed(1)
    random.shuffle(class_folder)
    return class_folder, class_folder[:train_size], class_folder[train_size:]