# 1D_conv_autoencoder
Using a 1D convolutional autoencoder to reduce the dimensionality of features extracted from paintings. Then I check if the visual similarities between the paintings are kept in the new space.

## Dataset
https://www.kaggle.com/ikarus777/best-artworks-of-all-time

# Description of the files 
1. The utils.py file contains some utils functions
2. The features_extraction.py extracts the interesting features from the paintings. A pre trained VGG16 CNN was used to compute the features, which were extracted from the fourth and fifth pooling layers of the network
3. The autoencoder.py file creates the architecture of the autoencoder and trains it. Five different models are trained. Check the doc.pdf file for more info on the working flow
4. The distance.py file taked the features files of reduced dimension and computes a silhouette score to see how separated are the clusters (represented by the paintings of an artist) in the new feature space.
