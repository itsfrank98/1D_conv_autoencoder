import os
import re
# import json
import numpy as np
# import json
# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.preprocessing import image
from tensorflow.keras.applications import VGG16  # , ResNet50
# from keras import layers
# from keras import Sequential
# from keras import optimizers
from keras import models
# from keras.callbacks import EarlyStopping

# from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------

### Monet vs Manet --> very similar painters


# function for file sorting
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


# load data
paths = []
# Monet
data_paths = ["dataset/images/images/Claude_Monet",
              "dataset/images/images/Edouard_Manet"]



for data_path in data_paths:
    for artwork in sorted_alphanumeric(os.listdir(data_path)):
        artwork_path = os.path.join(data_path, artwork).replace("\\", "/")
        if os.path.isfile(artwork_path):
            paths.append(artwork_path)

'''
# display an image
x = paths[0]
img = mpimg.imread(x)
imgplot = plt.imshow(img)
plt.show()
'''

# preprocessing
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img /= 255.0
    return img

#X is the array of preprocessed images
X = []

for img_path in paths:
    x = preprocess_image(img_path)
    X.append(x)

X = np.asarray(X)

'''
# display a preprocessed image
img = X[0]
imgplot = plt.imshow(img)
plt.show()
'''
# ------------------------------------------------------------------------------

# labels
labels = np.zeros(len(paths))

for i in range(len(labels)):  # they are quite balanced
    if i >= 73:
        labels[i] = 1
print(labels)
# ------------------------------------------------------------------------------

# vgg-based feature extraction without fine-tuning
conv_net = VGG16(weights='imagenet',
                 include_top=False,
                 input_shape=(224, 224, 3))

conv_net.summary()

for layer in conv_net.layers:
    print(layer.name)

features = []

layer_name = 'block4_pool'
intermediate_model = models.Model(inputs=conv_net.input,
                                  outputs=conv_net.get_layer(layer_name).output)

for x in X:
    x = np.expand_dims(x, axis=0)
    single_features = intermediate_model.predict(x)
    shape = intermediate_model.output_shape
    features.append(single_features.reshape(1 * shape[1] * shape[2] * shape[3]))



# features_scaled = scale(features)

# ------------------------------------------------------------------------------

# pca embedding
pca = PCA(n_components=100, random_state=42)  # usually, 100 components explains 80% variance
pca_embedding = pca.fit_transform(features)

plt.scatter(*zip(*pca_embedding[:, :2]), c=labels, cmap='viridis')
plt.show()

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('# components')
plt.ylabel('Cumulative explained variance')

# pca_embedding_scaled = scale(pca_embedding)
# la normalizzazione appiattisce le differenze!
# occorre normalizzare la distanze euclidea?

# ------------------------------------------------------------------------------

# t-SNE visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_embedding = tsne.fit_transform(pca_embedding)

plt.scatter(*zip(*tsne_embedding[:, :2]), c=labels, cmap='viridis')
plt.show()
'''
# ------------------------------------------------------------------------------

# TODO: unsupervised clustering evaluation
# TODO: metric learning?

# ------------------------------------------------------------------------------

# nearest neighbor search
neigh = NearestNeighbors(n_neighbors=4)

all_neighbors_paths = []
all_neighbors = []
all_distances = []

neigh.fit(tsne_embedding)

for j in range(len(tsne_embedding)):
    rng = tsne_embedding[j].reshape(1, -1)
    neighbors = neigh.kneighbors(rng, return_distance=False)[0]
    distances = neigh.kneighbors(rng, return_distance=True)[0]

    single_artwork_neighbors = []

    for q in range(len(neighbors)):
        single_artwork_neighbors.append(paths[neighbors[q]])

    all_neighbors_paths.append(single_artwork_neighbors)
    all_neighbors.append(neighbors)
    all_distances.append(distances)

# ------------------------------------------------------------------------------

# plotting neighbors
vec = [0, 1, 2, 3]

for j in range(len(vec)):
    i = vec[j]

    artwork = paths[i]
    print("query:")

    img = mpimg.imread(artwork)
    imgplot = plt.imshow(img)
    plt.axis('off')
    plt.show()

    neighbors_paths = all_neighbors_paths[i]

    print("responses:")
    for k in range(1, len(neighbors_paths)):
        img = mpimg.imread(neighbors_paths[k])
        imgplot = plt.imshow(img)
        plt.axis('off')
        plt.show()

for j in range(len(vec)):
    i = vec[j]

    neighbors_paths = all_neighbors_paths[i]
    distances = all_distances[i]

    print("query:")
    print(paths[i].split('/')[12].split('.')[0])

    print("responses:")
    for k in range(1, len(neighbors_paths)):
        idx = string = neighbors_paths[k].split('/')[12].split('.')[0]
        #        u = pca_embedding[i]
        #        v = pca_embedding[all_neighbors[i][k]]
        #        distance = round(0.5 * np.var(u - v) / (np.var(u) + np.var(v)), 4)
        #        print(str(idx) + " --> " + str(distance))
        print(str(idx) + " --> " + str(distances[0][k])[:5])

    print()'''

"""
block1_pool

query:
Claude_Monet_1
responses:
Claude_Monet_71 --> 952.8
Claude_Monet_73 --> 975.6
Claude_Monet_3 --> 979.7

query:
Claude_Monet_2
responses:
Claude_Monet_17 --> 1129.
Claude_Monet_8 --> 1137.
Claude_Monet_52 --> 1146.

query:
Claude_Monet_3
responses:
Claude_Monet_60 --> 658.7
Edouard_Manet_52 --> 664.2
Claude_Monet_19 --> 693.0

query:
Claude_Monet_4
responses:
Claude_Monet_33 --> 1141.
Edouard_Manet_76 --> 1148.
Edouard_Manet_27 --> 1161.

-------------------------------------------------------------------------------

block2_pool

query:
Claude_Monet_1
responses:
Claude_Monet_17 --> 2783.
Claude_Monet_30 --> 2794.
Claude_Monet_63 --> 2839.

query:
Claude_Monet_2
responses:
Claude_Monet_47 --> 3085.
Claude_Monet_17 --> 3111.
Claude_Monet_8 --> 3131.

query:
Claude_Monet_3
responses:
Claude_Monet_73 --> 2021.
Claude_Monet_33 --> 2027.
Claude_Monet_60 --> 2048.

query:
Claude_Monet_4
responses:
Claude_Monet_17 --> 2976.
Claude_Monet_58 --> 2991.
Edouard_Manet_62 --> 3091.

-------------------------------------------------------------------------------

block3_pool

query:
Claude_Monet_1
responses:
Claude_Monet_6 --> 1182.
Claude_Monet_36 --> 1352.
Claude_Monet_60 --> 1412.

query:
Claude_Monet_2
responses:
Claude_Monet_47 --> 1918.
Claude_Monet_32 --> 1951.
Claude_Monet_25 --> 2038.

query:
Claude_Monet_3
responses:
Claude_Monet_73 --> 1458.
Claude_Monet_24 --> 1539.
Claude_Monet_33 --> 1587.

query:
Claude_Monet_4
responses:
Claude_Monet_36 --> 1761.
Claude_Monet_1 --> 1765.
Claude_Monet_6 --> 1773.
"""

# ------------------------------------------------------------------------------

# TODO: at which feature level different artists overlap?
# TODO: derive an average score for measuring the overlapping between artists
'''
# clustering evaluation
print(silhouette_score(tsne_embedding, labels, metric='euclidean'))
print(calinski_harabasz_score(tsne_embedding, labels))
print(davies_bouldin_score(tsne_embedding, labels))'''
"""
block1_pool
0.22113235
43.141013247324054
1.642992893865469

block2_pool
0.15359291
28.407264932193797
2.0533290108484534

block3_pool
0.14981993
25.514927400613576
2.143674104495106

block4_pool
0.10112555
10.341116005144901
3.355750595629735

block5_pool
0.26780313
62.545146276490364
1.3570702961776369
"""
'''

def plot_features(features):   #only for 64 shaped features
    # plot all 64 maps in an 8x8 squares
    square = 8
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(features[0, :, :, ix - 1], cmap='gray')
            ix += 1
    # show the figure
    plt.show()
'''