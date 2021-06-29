from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dense, Flatten, Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras import Model
from utils import unload_features

def build_autoencoder(input_dim):
    """
    Function that builds the autoencoder which will be used to reduce the dimension of the features arrays
    :param input_dim: Starting dimension of the data
    :return: (autoencoder, encoder)
        autoencoder is the complete autoencoder
        encoder is the encoder that will be later used to reduce the dimensionality of the features array
    """
    input_layer = Input(shape=(input_dim, 1))
    enc = Conv1D(filters=16, kernel_size=2, padding='same', activation='relu')(input_layer)
    enc = MaxPooling1D(pool_size=2, padding='same')(enc)
    enc = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(enc)
    enc = MaxPooling1D(pool_size=2, padding='same')(enc)
    enc = Conv1D(filters=64, kernel_size=2, padding='same', activation='relu')(enc)
    enc = MaxPooling1D(pool_size=2, padding='same')(enc)
    enc = Flatten()(enc)
    enc = Dense(64)(enc)

    dec = Dense(200704)(enc)
    dec = Reshape((3136, 64))(dec)
    dec = Conv1D(filters=64, kernel_size=2, padding='same', activation='relu')(dec)
    dec = UpSampling1D(2)(dec)
    dec = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(dec)
    dec = UpSampling1D(2)(dec)
    dec = Conv1D(filters=16, kernel_size=2, padding='same', activation='relu')(dec)
    dec = UpSampling1D(2)(dec)
    dec = Conv1D(filters=1, kernel_size=2, padding='same', activation='relu')(dec)

    autoencoder = Model(input_layer, dec)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    autoencoder.summary()
    encoder = Model(input_layer, enc)
    return autoencoder, encoder


def fit_model(autoencoder, encoder, data, epochs, batch_size, autoencoder_model_name, encoder_model_name):
    checkpoint_filepath = 'checkpoints/{epoch:02d}.hdf5'
    model_checkpoint_callback = ModelCheckpoint(
                    filepath=checkpoint_filepath,
                    save_weights_only=False,
                    monitor='accuracy',
                    mode='max',
                    verbose=1,
                    save_best_only=True)
    autoencoder.fit(data,
                    data,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    verbose=1,
                    callbacks=[EarlyStopping(monitor='loss', patience=20),
                               model_checkpoint_callback])
    autoencoder.save(autoencoder_model_name)
    encoder.save(encoder_model_name)


if __name__ == '__main__':
    ######### FEATURES 5 ###############
    dir = 'features/features_5_pool'
    data = unload_features(dir)
    print(data.shape)
    (autoencoder, encoder) = build_autoencoder(data.shape[1])
    fit_model(autoencoder, encoder, data, 200, 32, 'pool5_autoencoder.h5', 'pool5_encoder.h5')
    model = load_model('autoencoder.h5')
    encoder = load_model('pool5_model.h5')

    ######### FEATURES 4.1 ###############
    dir = 'Features_4_1_pool'
    data = unload_features(dir)
    (autoencoder, encoder) = build_autoencoder(data.shape[1])
    fit_model(autoencoder, encoder, data, 200, 32, 'pool4_1_autoencoder.h5', 'pool4_1_encoder.h5')

    ######### FEATURES 4.2 ###############
    dir = 'features/features_4_2_pool'
    data = unload_features(dir)
    (autoencoder, encoder) = build_autoencoder(data.shape[1])
    fit_model(autoencoder, encoder, data, 200, 32, 'pool4_2_autoencoder.h5', 'pool4_2_encoder.h5')

    ######### FEATURES 4.3 ###############
    dir = 'features/features_4_3_pool'
    data = unload_features(dir)
    (autoencoder, encoder) = build_autoencoder(data.shape[1])
    fit_model(autoencoder, encoder, data, 200, 32, 'pool4_3_autoencoder.h5', 'pool4_3_encoder.h5')

    ######### FEATURES 4.4 ###############
    dir = 'features/features_4_4_pool'
    data = unload_features(dir)
    (autoencoder, encoder) = build_autoencoder(data.shape[1])
    fit_model(autoencoder, encoder, data, 200, 32, 'pool4_4_autoencoder.h5', 'pool4_4_encoder.h5')

    '''
    model = load_model('pool4_1_autoencoder.h5')
    d = deserialize_features('features/features4_pool_2/Jackson_Pollock_2.npy')
    pred = model.predict(d)
    #print(pred.shape)
    print(d[0, :100])
    print(pred[0, :100])
    '''