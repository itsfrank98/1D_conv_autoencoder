from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from collections import defaultdict
import os
from utils import deserialize_features
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def create_cluster_artists(artist_dict: defaultdict):
    v = np.vstack([artist_dict[k] for k in artist_dict.keys()])     #Stack vertically the arrays belonging to an artist
    y = []      #labels
    for k in artist_dict.keys():
        y = y + ([k.split('.')[0]] * len(artist_dict[k]))
    return v, y


def create_cluster_movements(artist_dict, cluster_name):
    v = np.vstack([artist_dict[k] for k in artist_dict.keys()])
    y = [cluster_name] * v.shape[0]
    return v, y

def create_scatter_plot(v, y):
    """
    Create a scatter plot to represent in a bidimensional space the paintings.
    """
    pca = PCA(n_components=100, random_state=42)
    pca_embedding = pca.fit_transform(v)
    labels = [0 if y[i] == y[0] else 1 for i in range(len(y))]      # Convert textual labels to numeric labels, which
                                                            # is what the 'c' parameter of the scatter function expects
    y = list(set(y))
    scatter = plt.scatter(*zip(*pca_embedding[:, :2]), c=labels, cmap='BrBG')
    plt.legend(handles=scatter.legend_elements()[0], labels=y)
    plt.show()


if __name__ == '__main__':
    artists_dict = defaultdict()
    dir = 'merged'

    '''
    #With this cycle you check the similarities of every possible artists pairing
    artists = os.listdir(dir)
    l = []
    for i in range(len(artists)):
        artists_dict = defaultdict()
        ar1 = artists[i].split('.')[0]
        artists_dict[ar1] = deserialize_features(os.path.join(dir, ar1+'.npy'))
        j = i+1
        while j < len(artists):
            ar2 = artists[j].split('.')[0]
            artists_dict[ar2] = deserialize_features(os.path.join(dir, ar2 + '.npy'))
            (v, y) = create_cluster_artists(artists_dict)
            l.append((silhouette_score(v, y), ar1, ar2))
            artists_dict.pop(ar2)
            j += 1
    l.sort()
    print(l)        
    '''
    '''
    #Cycle to compare couples of artists
    artists = ['Vincent_van_Gogh', 'Titian']
    for artist in artists:
        artists_dict[artist] = deserialize_features(os.path.join(dir, artist+'.npy'))
    (v, y) = create_cluster_artists(artists_dict)

    '''
    #Comparison between artistic movements
    renaissance_artists = ['Sandro_Botticelli', 'Titian', 'Leonardo_da_Vinci', 'Michelangelo', 'Raphael']
    impressionism_artists = ['Claude_Monet', 'Edouard_Manet', 'Camille_Pissarro', 'Edgar_Degas', 'Pierre-Auguste_Renoir']
    artist_dict_ren = defaultdict()
    artist_dict_impr = defaultdict()
    
    for artist in renaissance_artists:
        artist_dict_ren[artist] = deserialize_features(os.path.join(dir, artist+'.npy'))

    for artist in impressionism_artists:
        artist_dict_impr[artist] = deserialize_features(os.path.join(dir, artist+'.npy'))
    
    (v0, y0) = create_cluster_movements(artist_dict_ren, 'Renaissance')
    (v1, y1) = create_cluster_movements(artist_dict_impr, 'Impressionism')
    v = np.vstack([v0, v1])
    y = y0 + y1

    create_scatter_plot(v, y)
    print(davies_bouldin_score(v, y))
    print(calinski_harabasz_score(v, y))
    print(silhouette_score(v, y))
