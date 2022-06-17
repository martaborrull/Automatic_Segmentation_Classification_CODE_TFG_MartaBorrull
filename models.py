#%% Cell 1
import tensorflow.keras.layers as Layers
from tensorflow import keras
from keras.models import Model
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16, VGG19

#%% Cell 2
#Initialize the parameters of the u-net model.

#Example of image size:
img_width = 256 #512 #256 #572
img_height = 256 #512 #256 #572
img_channels = 1 #1 if the picture is in grayscale, 3 if it is an RGB picture


def Unet_model(img_width, img_length, img_channels):
    Inputs = Layers.Input((img_width,img_height,img_channels))

         
    
    #Added dropout to the model which seems to improve performance.

    #forward
    c1_1 = Layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu', kernel_initializer="he_normal",padding = 'same')(Inputs)
    c1_1 = Layers.BatchNormalization()(c1_1)
    c1_2 = Layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu', kernel_initializer="he_normal",padding = 'same')(c1_1)
    c1_2 = Layers.BatchNormalization()(c1_2)
    drop1 = Layers.Dropout(0.2)(c1_2)
    maxpool_1 = Layers.MaxPool2D(pool_size = (2,2))(drop1)

    c2_1 = Layers.Conv2D(filters = 128, kernel_size = 3, activation = 'relu', kernel_initializer="he_normal",padding = 'same')(maxpool_1)
    c2_1 = Layers.BatchNormalization()(c2_1)
    c2_2 = Layers.Conv2D(filters = 128, kernel_size = 3, activation = 'relu', kernel_initializer="he_normal",padding = 'same')(c2_1)
    c2_2 = Layers.BatchNormalization()(c2_2)
    drop2 = Layers.Dropout(0.2)(c2_2)
    maxpool_2 = Layers.MaxPool2D(pool_size = (2,2))(drop2)

    c3_1 = Layers.Conv2D(filters = 256, kernel_size = 3, activation = 'relu', kernel_initializer="he_normal",padding = 'same')(maxpool_2)
    c3_1 = Layers.BatchNormalization()(c3_1)
    c3_2 = Layers.Conv2D(filters = 256, kernel_size = 3, activation = 'relu', kernel_initializer="he_normal",padding = 'same')(c3_1)
    c3_2 = Layers.BatchNormalization()(c3_2)
    drop3 = Layers.Dropout(0.2)(c3_2)
    maxpool_3 = Layers.MaxPool2D(pool_size = (2,2))(c3_2)

    c4_1 = Layers.Conv2D(filters = 512, kernel_size = 3, activation = 'relu', kernel_initializer="he_normal",padding = 'same')(maxpool_3)
    c4_1 = Layers.BatchNormalization()(c4_1)
    c4_2 = Layers.Conv2D(filters = 512, kernel_size = 3, activation = 'relu', kernel_initializer="he_normal",padding = 'same')(c4_1)
    c4_2 = Layers.BatchNormalization()(c4_2)
    drop4 = Layers.Dropout(0.2)(c4_2)
    maxpool_4 = Layers.MaxPool2D(pool_size = (2,2))(c4_2)

    c5_1 = Layers.Conv2D(filters = 1024, kernel_size = 3, activation = 'relu', kernel_initializer="he_normal",padding = 'same')(maxpool_4)
    c5_1 = Layers.BatchNormalization()(c5_1)
    c5_2 = Layers.Conv2D(filters = 1024, kernel_size = 3, activation = 'relu', kernel_initializer="he_normal",padding = 'same')(c5_1)
    c5_2 = Layers.BatchNormalization()(c5_2)
    drop5 = Layers.Dropout(0.2)(c5_2)
    
    #backward
    up6 = Layers.UpSampling2D(size = (2,2))(drop5) 
    #convolution + up-sampling layer
    u_c6 = Layers.Conv2D(filters = 512, kernel_size = 2, activation = 'relu', kernel_initializer="he_normal",padding = 'same')(up6) 
    u_c6 = Layers.concatenate([u_c6,drop4])
    c6_1 = Layers.Conv2D(filters = 512, kernel_size = 3, activation = 'relu', kernel_initializer="he_normal",padding = 'same')(u_c6)
    c6_1 = Layers.BatchNormalization()(c6_1)
    c6_2 = Layers.Conv2D(filters = 512, kernel_size = 3, activation = 'relu', kernel_initializer="he_normal",padding = 'same')(c6_1)
    c6_2 = Layers.BatchNormalization()(c6_2)
    
    up7 = Layers.UpSampling2D(size = (2,2))(c6_2)
    u_c7 = Layers.Conv2D(filters = 256, kernel_size = 2, activation = 'relu', kernel_initializer="he_normal",padding = 'same')(up7)
    u_c7 = Layers.concatenate([u_c7,drop3])
    c7_1 = Layers.Conv2D(filters = 256, kernel_size = 3, activation = 'relu', kernel_initializer="he_normal",padding = 'same')(u_c7)
    c7_1 = Layers.BatchNormalization()(c7_1)
    c7_2 = Layers.Conv2D(filters = 256, kernel_size = 3, activation = 'relu', kernel_initializer="he_normal",padding = 'same')(c7_1)
    c7_2 = Layers.BatchNormalization()(c7_2)

    up8 = Layers.UpSampling2D(size = (2,2))(c7_2)
    u_c8 = Layers.Conv2D(filters = 128, kernel_size = 2, activation = 'relu', kernel_initializer="he_normal",padding = 'same')(up8)
    u_c8 = Layers.concatenate([u_c8,drop2])
    c8_1 = Layers.Conv2D(filters = 128, kernel_size = 3, activation = 'relu', kernel_initializer="he_normal",padding = 'same')(u_c8)
    c8_1 = Layers.BatchNormalization()(c8_1)
    c8_2 = Layers.Conv2D(filters = 128, kernel_size = 3, activation = 'relu', kernel_initializer="he_normal",padding = 'same')(c8_1)
    c8_1 = Layers.BatchNormalization()(c8_1)

    up9 = Layers.UpSampling2D(size = (2,2))(c8_2)
    u_c9 = Layers.Conv2D(filters = 64, kernel_size = 2, activation = 'relu', kernel_initializer="he_normal",padding = 'same')(up9)
    u_c9 = Layers.concatenate([u_c9,drop1])
    c9_1 = Layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu', kernel_initializer="he_normal",padding = 'same')(u_c9)
    c9_1 = Layers.BatchNormalization()(c9_1)
    c9_2 = Layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu', kernel_initializer="he_normal",padding = 'same')(c9_1)
    c9_2 = Layers.BatchNormalization()(c9_2)
    c9_3 = Layers.Conv2D(filters = 2, kernel_size = 1)(c9_2)
    c9_3 = Layers.BatchNormalization()(c9_3)
    
    output = Layers.Conv2D(filters = 1, kernel_size = 1, activation = 'sigmoid')(c9_3)


    return Model(inputs = Inputs, outputs = output)

#%% Cell 3

input_shape = (img_width,img_height,img_channels)

Inputs = Layers.Input(input_shape)
print(Inputs)


def mod_vgg16_model(Inputs):
    vgg16_model = VGG16(weights=None , include_top=False, input_tensor=Inputs)
    
    for layer in vgg16_model.layers:
        layer.trainable = False
    
    model = keras.models.Sequential([
    vgg16_model,
    Layers.Flatten(),
    Layers.Dense(64, activation='relu'),
    Layers.Dense(64, activation='relu'),
    Layers.Dense(2, activation='sigmoid')])
    
    return model

#%% Cell 4    

def mod_vgg19_model(Inputs):
    vgg16_model = VGG19(weights=None , include_top=False, input_tensor=Inputs)
    
    for layer in vgg16_model.layers:
        layer.trainable = False
    
    model = keras.models.Sequential([
    vgg16_model,
    Layers.Flatten(),
    Layers.Dense(64, activation='relu'),
    Layers.Dense(64, activation='relu'),
    Layers.Dense(2, activation='sigmoid')])
    
    return model





#%% Plot the loss_accuracy

def plot_loss_acc(model_fitted):
    # loss
    plt.plot(model_fitted.history['loss'], label='train loss')
    plt.plot(model_fitted.history['val_loss'], label='val loss')
    plt.legend()
    plt.show()
    
    # accuracies
    plt.plot(model_fitted.history['accuracy'], label='train acc')
    plt.plot(model_fitted.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.show()