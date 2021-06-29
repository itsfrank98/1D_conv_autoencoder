from keras.models import load_model
from utils import serialize_features, deserialize_features
import os
from tqdm import tqdm
import numpy as np

def reduce_dimensionality(src_dir, model, dst_dir):
    """
    Takes the features arrays and reduces their dimensionality by using the model. The reduced features are serialized
    :param src_dir: Path to the directory containing the features to reduce
    :param model: Model to use to reduce the features dimensionality
    :param dst_dir: Path to the folder where the reduced features will be saved
    """
    desc = src_dir.split('/')[1]    #The progress bar will show the name of the current src directory
    for f in tqdm(os.listdir(src_dir), desc=desc):
        features = deserialize_features(os.path.join(src_dir, f))
        reduced_features = model.predict(features)

        file_name = f.split('.')[0]
        if desc != "features_5_pool":
            file_name = file_name[:-2]      #We want to serialize the reduced arrays without the index number at the end

        serialize_features(reduced_features, os.path.join(dst_dir, file_name))

def merge_features_vectors(src_directories, dst_directory):
    """
    For each artist, takes the reduced feature vectors compued for that artist (four for every artist) and merges them,
    serializing the results in an other directory. The order is pool4_1, pool4_2, pool4_3, pool4_4, pool5
    :param src_directories: List of directories containing the feature files
    :param dst_directory:
    :return:
    """
    artists = os.listdir(os.path.join(src_directories[0]))

    for artist in artists:
        vec = np.hstack([deserialize_features(os.path.join(src_directories[i], artist)) for i in range(len(src_directories))])
        print(vec.shape)
        serialize_features(vec, os.path.join(dst_directory, artist.split('.')[0]))


if __name__ == '__main__':
    features_dirs = ['features/features_4_1_pool', 'features/features_4_2_pool', 'features/features_4_3_pool',
                     'features/features_4_4_pool', 'features/features_5_pool']

    reduced_features_dirs = ['reduced/reduced_4_1_pool', 'reduced/reduced_4_2_pool', 'reduced/reduced_4_3_pool',
                             'reduced/reduced_4_4_pool', 'reduced/reduced_5_pool', ]

    models = ['models/pool4_1_encoder.h5', 'models/pool4_2_encoder.h5', 'models/pool4_3_encoder.h5',
              'models/pool4_4_encoder.h5', 'models/pool5_encoder.h5']

    for i in range(len(features_dirs)):
        src_dir = features_dirs[i]
        dst_dir = reduced_features_dirs[i]
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        model = load_model(models[i], compile=False)
        reduce_dimensionality(src_dir, model, dst_dir)

    merged_features_dir = 'merged'

    if not os.path.exists(merged_features_dir):
        os.makedirs(merged_features_dir)
    merge_features_vectors(reduced_features_dirs, merged_features_dir)
