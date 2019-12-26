#!/usr/bin/env python
# coding: utf-8

# In[10]:


from emnist import list_datasets
from matplotlib import pyplot as plt
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Conv1D
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.regularizers import l2
from keras.losses import mse, binary_crossentropy
list_datasets()


# In[11]:


import data_prepro as preprocessing


def noise_addition(data, noise_factor):
    data_noisy = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape) 
    return data_noisy


# In[12]:
train_images, train_labels, test_images, test_labels=preprocessing.input_data()

#train_images, train_labels, test_images, test_labels=input_data()
train_images_noisy=noise_addition(train_images,0.1)
test_images_noisy= noise_addition(test_images,0.1)
x_train_noisy = np.clip(train_images_noisy, 0., 1.)
x_test_noisy = np.clip(test_images_noisy, 0., 1.)
n = 10
plt.figure(figsize=(20, 2))
for i in range(1,n):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

x_train_noisy = np.reshape(x_train_noisy, (len(x_train_noisy), 28, 28, 1))
x_test_noisy = np.reshape(x_test_noisy, (len(x_test_noisy), 28, 28, 1))


# In[25]:


def sampling1(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal_variable(shape=(100, 2), mean=0.,scale=1.0)
    return z_mean + K.exp(z_log_var/2.0) * epsilon
def plot1(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot2(test_images, vae_eval,x_test_noisy):
    def show_side_by_side(left,middle, right):
        zipped = zip(left,middle,right)
        num_rows = min(len(right), len(left))
        f, axes = plt.subplots(num_rows, 3, sharex=True, sharey=True,figsize=(15,15))

        n_row = 0
        for l,m, r in zipped:
            axes[n_row,0].imshow(l)
            axes[n_row,1].imshow(m)
            axes[n_row,2].imshow(r)
            n_row += 1
        f.tight_layout()
        plt.show()

    def evaluate(X, vae,noisy_data):
        X=X.reshape(10, 28, 28,1)
        out_example = vae.predict(X[:10])
        noisy_data=X
        noisy_data=noisy_data.reshape(10, 28, 28)
        out_example=out_example.reshape(10, 28, 28)
        show_side_by_side(test_images,noisy_data, out_example)
    evaluate(test_images[:10], vae_eval,x_test_noisy)
    
def plot3(encoder):
    z_mean, _= encoder.predict(x_test_noisy, batch_size=100)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=test_labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()
    
def plot4(decoder):
    n = 10
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-2, 2, n)
    grid_y = np.linspace(-2, 2, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit
    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.show()
    
    
def Architecture_1_Variation_AE():
    image_size = 28
    input_shape = (image_size, image_size, 1)
    original_dim_flat = np.prod(input_shape)
    latent_dim = 2
    intermediate_dim = 512
    inputs = Input(shape=input_shape, name='input_encoder')
    f = Flatten(name="flat_incoder")(inputs)
    h = Dense(intermediate_dim, activation="relu", name="encoding_dense")(f)
    z_mean = Dense(latent_dim, activation="sigmoid", name="mean")(h)
    z_log_var = Dense(latent_dim, name="dense_var")(h)
    z = Lambda(sampling1, output_shape=(latent_dim,), name="sampling")([z_mean, z_log_var])
    encoder = Model(inputs, [z, z_mean], name="encoder")
    inp_encoded = Input(shape=(latent_dim,), name="input_decoder")
    decoded = Dense(intermediate_dim, activation="relu", name="decoding_Dense")(inp_encoded)
    flat_decoded = Dense(original_dim_flat, activation="sigmoid", name="flat_decoder")(decoded)
    out_decoded = Reshape(input_shape, name="output_decoder")(flat_decoded)
    decoder = Model(inp_encoded, out_decoded, name="decoder")
    out_train = decoder(encoder(inputs)[0])
    out_eval = decoder(encoder(inputs)[1])
    train = Model(inputs, out_train)
    evalu = Model(inputs, out_eval)
    train.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return train, evalu,decoder,encoder

def Architecture_2_Variation_AE():
    image_size=28
    input_shape = (image_size, image_size, 1)
    original_dim_flat = np.prod(input_shape)
    latent_dim = 2
    inputs = Input(shape=input_shape, name='input_encoder')
    x1 = Conv2D(16, (3, 3), activation='relu', padding='same',name="encoder_conv1")(inputs)
    x2 = MaxPooling2D((2, 2), padding='same',name="encoder_maxpool")(x1)
    x3 = Conv2D(8, (3, 3), activation='relu', padding='same',name="encoder_conv2")(x2)
    x4 = MaxPooling2D((2, 2), padding='same')(x3)
    x5 = Flatten(name="flatten")(x4)
    z_mean = Dense(latent_dim, activation="sigmoid", name="mean")(x5)
    z_log_var = Dense(latent_dim, name="dense_var")(x5)
    z = Lambda(sampling1, output_shape=(latent_dim,), name="sampling")([z_mean, z_log_var])
    encoder = Model(inputs, [z, z_mean], name="input_decoder")

    inp_encoded = Input(shape=(latent_dim,), name="decoding_input")
    decoded = Dense(512,kernel_regularizer=l2(1e-4), activation="relu", name="decoding_dense")(inp_encoded)
    decoded1 = Dense(256,kernel_regularizer=l2(1e-4), activation="relu", name="decoding_dense2")(decoded)
    flat_decoded = Dense(original_dim_flat, activation="sigmoid", name="flat_decoder")(decoded1)
    out_decoded = Reshape(input_shape, name="reshape_decoded")(flat_decoded)
    decoder = Model(inp_encoded, out_decoded, name="decoder")
    out_train = decoder(encoder(inputs)[0])
    out_eval = decoder(encoder(inputs)[1])
    train = Model(inputs, out_train)
    evalu = Model(inputs, out_eval)
    train.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return train, evalu,decoder,encoder

def Architecture_1_VAE():
    input_img = Input(shape=(28, 28, 1))
    x1 = Conv2D(16, (3, 3), activation='relu', padding='same',name="encoder_conv1")(input_img)
    x2 = MaxPooling2D((2, 2), padding='same',name="encoder_maxpool1")(x1)
    x3 = Conv2D(8, (3, 3), activation='relu', padding='same',name="encoder_conv2")(x2)
    x4 = MaxPooling2D((2, 2), padding='same',name="encoder_maxpool2")(x3)
    x5 = Conv2D(8, (3, 3), activation='relu', padding='same',name="encoder_conv3")(x4)
    encoder = MaxPooling2D((2, 2), padding='same')(x5)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoder = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_img, decoder)
    autoencoder.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    history= autoencoder.fit(x_train_noisy, x_train_noisy,epochs=2,batch_size=128,shuffle=True,validation_data=(x_test_noisy, x_test_noisy))
    return history,autoencoder,encoder,decoder

def Architecture_3_Variation_AE():
    image_size=28
    input_shape = (image_size, image_size, 1)
    intermediate_dim = 512
    latent_dim = 2
    inputs = Input(shape=input_shape, name='input_encoder')
    x1 = Conv2D(16, (3, 3), activation='relu', padding='same',name="encoder_conv1")(inputs)
    x2 = MaxPooling2D((2, 2), padding='same',name="encoder_maxpool1")(x1)
    x3 = Conv2D(8, (3, 3), activation='relu', padding='same',name="encoder_conv2")(x2)
    x4 = MaxPooling2D((2, 2), padding='same',name="encoder_maxpool2")(x3)
    x5 = Conv2D(8, (3, 3), activation='relu', padding='same',name="encoder_conv3")(x4)
    x6 = MaxPooling2D((2, 2), padding='same',name="encoder_maxpool3")(x5)
    shape = K.int_shape(x6)
    x7 = Flatten()(x6)
    z_mean = Dense(latent_dim, name='mean')(x7)
    z_log_var = Dense(latent_dim, name='sampling')(x7)
    z = Lambda(sampling1, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z, z_mean], name='encoder')
    inp_encoded = Input(shape=(latent_dim,), name='decoding_input')
    x1 = Dense(shape[1] * shape[2] * shape[3], activation='relu',name="decoder_dense1")(inp_encoded)
    x2 = Reshape((shape[1], shape[2], shape[3]),name="decoder_reshape")(x1)
    x3 = Conv2D(8, (3, 3), activation='relu', padding='same',name="decoder_conv1")(x2)
    x4 = UpSampling2D((2, 2),name="decoder_upsammpling1")(x3)
    x5 = Conv2D(8, (3, 3), activation='relu', padding='same',name="decoder_conv2")(x4)
    x6 = UpSampling2D((2, 2),name="decoder_upsammpling2")(x5)
    x7 = Conv2D(16, (3, 3), activation='relu',name="decoder_conv3")(x6)
    x8 = UpSampling2D((2, 2),name="decoder_upsammpling3")(x7)
    outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same',name="decoder_conv4")(x8)
    decoder = Model(inp_encoded, outputs, name='decoder')
    out_train = decoder(encoder(inputs)[0])
    out_eval = decoder(encoder(inputs)[1])
    train = Model(inputs, out_train)
    evalu = Model(inputs, out_eval)
    
    train.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return train, evalu,decoder,encoder


# In[27]:


train1, evalu1 ,decoder1,encoder1 = Architecture_1_Variation_AE()
print("\n" + "-" * 100 + "\nEncoder\n" + "-" * 100)
train1.layers[1].summary()
print("\n" + "-" * 100 + "\nDecoder\n" + "-" * 100)
train1.layers[2].summary()
print("\n" + "-" * 100 + "\n" + "VAE\n" + "-" * 100)
train1.summary()

train2, evalu2 ,decoder2,encoder2 = Architecture_2_Variation_AE()
print("\n" + "-" * 100 + "\nEncoder\n" + "-" * 100)
train2.layers[1].summary()
print("\n" + "-" * 100 + "\nDecoder\n" + "-" * 100)
train2.layers[2].summary()
print("\n" + "-" * 100 + "\n" + "VAE\n" + "-" * 100)
train2.summary()

train3, evalu3,decoder3,encoder3= Architecture_3_Variation_AE()
print("\n" + "-" * 100 + "\nEncoder\n" + "-" * 100)
train3.layers[1].summary()
print("\n" + "-" * 100 + "\nDecoder\n" + "-" * 100)
train3.layers[2].summary()
print("\n" + "-" * 100 + "\n" + "VAE\n" + "-" * 100)
train3.summary()


# In[24]:


history1=train1.fit(x_train_noisy, x_train_noisy, shuffle=True, epochs=2, batch_size=100, validation_data=(x_test_noisy,x_test_noisy), verbose=1)
plot1(history1)
plot2(test_images[:10], evalu1,x_test_noisy)
plot3(encoder1)
plot4(decoder1)
history2=train2.fit(x_train_noisy, x_train_noisy, shuffle=True, epochs=2, batch_size=100, validation_data=(x_test_noisy,x_test_noisy), verbose=1)
plot1(history2)
plot2(test_images[:10], evalu2,x_test_noisy)
plot3(encoder2)
plot4(decoder2)
history3=train3.fit(x_train_noisy, x_train_noisy, shuffle=True, epochs=2, batch_size=100, validation_data=(x_test_noisy,x_test_noisy), verbose=1)
plot1(history3)
plot2(test_images[:10], train3,x_test_noisy)
plot3(encoder3)
plot4(decoder3)
history4,autoencoder4,encoder4,decoder4=Architecture_1_VAE()
plot1(history4)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




