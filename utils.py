import os
import numpy as np
from collections import defaultdict
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from tqdm import tqdm

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)   #expand the array from one in 3D to one in 4D (samples, rows, cols, channels)
    img = preprocess_input(img)
    return img

def build_artists_dictionary(artists, paths):
    d = defaultdict(list)  # dict whose keys are the names of the artists, values are the paths to their paintings
    for img_path in paths:
        for ar in artists:
            if img_path.__contains__(ar):
                break
        x = preprocess_image(img_path)
        d[ar].append(x)
    return d

def build_images_paths(paths_to_directories):
    """
    Builds the paths to reach every image in a directory.
    :param paths_to_directories: List containing the directories from which we want to load images
    :return:
    """
    paths_to_images = []
    for path in paths_to_directories:
        for artwork in os.listdir(path):
            artwork_path = os.path.join(path, artwork).replace("\\", "/")
            if os.path.isfile(artwork_path):
                paths_to_images.append(artwork_path)
    return paths_to_images

def serialize_features(features_array, path_to_file):
    """
    Save a features list as a npy file
    :param features_list: List to save
    :param path_to_file: Path to the file (included its name)
    """
    try:
        os.remove(path_to_file)
    except OSError:
        pass
    np.save(path_to_file, features_array)

def deserialize_features(path_to_file):
    """
    Deserialize one single features file
    :param path_to_file:
    :return:
    """
    features_list = np.load(path_to_file, allow_pickle=True)
    return features_list

def unload_features(directory):
    """
    Deserialize all the features files in a directory and put them all in an array. The arrays are concatenated
    vertically
    :param directory: source directory containing the arrays to stack
    :return: stacked array
    """
    artists = os.listdir(os.path.join(directory))
    return np.vstack([deserialize_features(os.path.join(directory, a)) for a in artists])

def split_features_files(path_to_directory, dst1, dst2, dst3, dst4):
    """
    Since the features coming from the pool4 layer have a dimensionality that is too large, we decide to split them in
    four equal parts, each of whose will be used to train a model
    :param path_to_directory: path to the directory containing the features to split
    :param dst1, dst2, dst3, dst4: Paths to the four directories in which the divided features will be put
    """
    for f in tqdm(os.listdir(path_to_directory)):
        features_matrix = deserialize_features(os.path.join(path_to_directory, f))
        split_point = int(features_matrix.shape[1]/4)
        np.save(os.path.join(dst1, f.split('.')[0]+'_1'), features_matrix[:, 0:split_point])
        np.save(os.path.join(dst2, f.split('.')[0]+'_2'), features_matrix[:, split_point:split_point*2])
        np.save(os.path.join(dst3, f.split('.')[0]+'_3'), features_matrix[:, split_point*2:split_point*3])
        np.save(os.path.join(dst4, f.split('.')[0]+'_4'), features_matrix[:, split_point*3:])


if __name__ == '__main__':
    dst1 = 'Features4_pool_1'
    dst2 = 'Features4_pool_2'
    dst3 = 'Features4_pool_3'
    dst4 = 'Features4_pool_4'
    path_to_directory = 'Features_4pool'
    split_features_files(path_to_directory, dst1, dst2, dst3, dst4)
