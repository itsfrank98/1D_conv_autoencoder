import tensorflow as tf
from collections import defaultdict
from tqdm import tqdm
from keras.applications.vgg16 import VGG16
from keras.models import Model
from utils import *

def extract_features_from_img(net: VGG16, layers_to_extract: list, img):
    """
    Extract the net features from the desired layers for one painting
    :param layers_to_extract: List containing the names of the layers from which extract features
    :param img: Matrix representing the image whose features will be extracted
    :param net: Neural network from which the features will be extracted
    :return:
        features: np.ndarray(): Mono dimensional array containing all the features extracted for the image from the
        specified layers of the net
    """
    features = np.empty(0)
    for layer_name in layers_to_extract:
        model = Model(inputs=net.inputs,
                      outputs=net.get_layer(layer_name).output)

        #Here I call model(img) instead of model.predict(img) since the latter is much slower if called in a loop
        feat = model(img)       #Feat is a Multidimensional eager tensor (4d)

        (d1, d2, d3, d4) = feat.shape
        feat = tf.reshape(feat, [d1 * d2 * d3 * d4])    #Reshape the 4D array in a 1D array

        features = np.concatenate((features, feat))      #Concatenate the newly obtained feature vector and the vector
                                                         # containing all the features for the image
    return features

def extract_features_from_artists(net, layers_to_extract, d: dict, dst, n):
    """
    Function that, for each artist, extracts an array containing the arrays of the features extracted for each painting
    from the specified layers of the convolutional net. Then saves the array a separate file. Therefore we will have one
    feature file for each artist.
    :param net: CNN from which the features will be extracted
    :param layers_to_extract: List of layers of the net from which the features will be extracted
    :param d: Dictionary containing, for every artist, the list of matrices of his paintings
    :param dst: Directory where the feature files will be serialized
    """
    #Righe = numero quadri, colonne = numero features
    for artist in d.keys():
        features_l = np.empty((len(d[artist]), n))
        i = 0
        for img in tqdm(d[artist], desc=artist):
            features_l[i] = (extract_features_from_img(net, layers_to_extract, img))
            i += 1
        serialize_features(features_l, os.path.join(dst, artist))


if __name__ == '__main__':
    img_dir = "dataset/images/images"
    artists = os.listdir(img_dir)
    img_paths = []
    features_dst_dir = 'features/features_5_pool'   #directory where the features will be put (one file for each artist)

    for a in artists:
        img_paths.append(img_dir + "/{}".format(a))

    #img_paths = [img_paths[12]]  #Delete here to load the full dataset
    #artists = [artists[0]]
    d = build_artists_dictionary(artists, img_paths)

    conv_net = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    layers_to_extract = ['block4_pool', 'block5_pool']

    #check if the specified layers exist in the network
    layers_names = [layer.name for layer in conv_net.layers]
    for layer in layers_to_extract:
        if layer not in layers_names:
            raise Exception("The network does not have a layer named {}".format(layer))

    if not os.path.exists(features_dst_dir):
        os.makedirs(features_dst_dir)
    #extract_features_from_artists(conv_net, layers_to_extract, d, features_dst_dir, 326144)  #345
    #extract_features_from_artists(conv_net, layers_to_extract, d, features_dst_dir, 125440)  #45
    #extract_features_from_artists(conv_net, layers_to_extract, d, features_dst_dir, 25088)   #5
    extract_features_from_artists(conv_net, layers_to_extract, d, features_dst_dir, 100352)  # 4
