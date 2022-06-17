#Marta Borrull Luque
#TFG: Automatic Segmentation and Classification for Pancreatitis
#Curs 2021-2022
#Director: Miguel Ángel Cordobés Aranda
#%%Cell 1

#Check if the code will run with the GPU
import platform
sistema = platform.system()
if sistema == 'Windows':
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    import tensorflow as tf
    print(tf.__version__)

    from keras import backend as K
    config = tf.compat.v1.ConfigProto()

    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    # Only allow a total of 40% the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.4


    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)


#%%Cell 2

#import the files where are defined the functions for 
#the models and the multiple functions used in the main file.

from models import *
from utils import *

#variables:
epochs = 50 
image_size = 256   
epochs_vgg = 20

#%%Cell 3

#Call the U-net model created in models.py
unet_model = Unet_model(img_width, img_height, img_channels)
unet_model.summary()


#%%Cell 4

#Select the location of the files, depending on the Operating System.

if sistema == 'Windows':
    print(sistema)
    path = 'F:/Archivos JPG/Dataset'
    dataset_path = 'F:/Archivos JPG/Dataset/All'
    test_path = 'F:/Archivos JPG/Dataset/Test'
    pred_path = 'F:/Archivos JPG/Dataset/Preds'
else:
    print(sistema)
    path = '/Volumes/TOSHIBA Ext/Archivos JPG/Dataset'
    dataset_path = '/Volumes/TOSHIBA Ext/Archivos JPG/Dataset/All'
    test_path = '/Volumes/TOSHIBA Ext/Archivos JPG/Dataset/Test'
    pred_path = '/Volumes/TOSHIBA Ext/Archivos JPG/Dataset/Preds'

#%%Cell 5

#Call the function that read the images from the file.
repeated_images, repeated_masks = data_repetition(dataset_path, None)


#%%Cell 6

#Prepare the name of the labels.

class_names = ['Pancreatitis','Normal']
class_names_label = {}
for i, name in enumerate(class_names):
    
    class_names_label[name] = i
    
num_classes = len(class_names_label)

print(class_names_label)

#%%Cell 7

#Call the function that load and prepare the images.
Images, Labels , Masks, Multiplied_images = data_preparation(repeated_images, repeated_masks, 256, class_names)

#%%Cell 8

#Check the first 10 images and their masks.
for i in range(10):
    x = np.squeeze(Images[i])
    y = np.squeeze(Masks[i]).astype(np.float32)

    show_images(x, y, None, None)

#%%Cell 9

#Split the whole data into train and test.

x_train_images, x_train_mult_images, y_train_labels, y_train_masks, x_test_images,x_test_mult_images, y_test_labels, y_test_masks = split_train_test(Images, Labels, Masks, Multiplied_images) 

#%%Cell 10

#Compile the u-net model and train it with the images and the masks.

unet_model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

segmentation = unet_model.fit(x_train_images, y_train_masks, validation_data=(x_test_images,y_test_masks),  batch_size = 8, epochs=epochs)

plot_loss_acc(segmentation)


#%%Cell 11

#Read the test files.
test_list = os.listdir(test_path)
test_path_list = []
num_pa = []

reject = ['Pancreas', 'Masks']
for path, subdirs, files in os.walk(test_path):
    if 'P_' in path:
        x = path[-5:]
        num_pa.append(x)
        pacient = []
        for image in files:
            if '._' not in image:
                pacient.append(os.path.join(path, image))
        test_path_list.append(pacient)
        
test_path_list = sorted(test_path_list)
image_shape = (image_size, image_size)

Xtest = []
for pacients in test_path_list:
    pacient = []
    for image in pacients:
        try:
            original_image = load_img(image, color_mode='grayscale')
        except PIL.UnidentifiedImageError:
            print(image)
            
        img = image_preparation(original_image, image_shape)
        pacient.append(img)
        
    Xtest.append(np.asarray(pacient,dtype=np.float32))
        
if len(Xtest) != 1:
    for pacient in Xtest:
        print('Xtest ', len(pacient))
else:
   print('Xtest ', len(Xtest))
    

print(num_pa)
#%%Cell 12

#Make the prediction of the test images.

predicted_images = []
pred_mult_images = []
for pacient in range(len(test_path_list)):
    preds = unet_model.predict(Xtest[pacient])
    multiplied_images = show_preds(Xtest[pacient], preds, pred_path,test_path_list[pacient])
    predicted_images.append(preds)
    pred_mult_images.append(multiplied_images)
    

#%%Cell 13

#We call this function to create an image to see the whole pancreas.
complete_pancreas = []
for pacient in predicted_images:
    complete_pancreas.append(total_pancreas(pacient))

#%%Cell 14

#Call the function that gives the area ratio.
for pancreas in complete_pancreas:
    area_pancreas(pancreas, image_size)


#%%Cell 15

#Call the pre-defined function of VGG16 and compile it.
vgg16_model = mod_vgg16_model(Inputs)
vgg16_model.summary()

vgg16_model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

#%%Cell 16

#Train the VGG 16 model with the multiplication images and labels.

classification_vgg16 = vgg16_model.fit(x_train_mult_images, y_train_labels, validation_data=(x_test_mult_images,y_test_labels), batch_size = 8, epochs=epochs_vgg)

plot_loss_acc(classification_vgg16)

#%%Cell 17

#Predict the vgg16 model with the test multiplication images calculated above.

pred_list_16 = []
for pacient in pred_mult_images:
    preds_clas = vgg16_model.predict(np.asarray(pacient, np.float32))
    print(preds_clas)
    pred_list_16.append(preds_clas)

    

#%%Cell 18

#Make the mean of all the patients

list_pred_means_16 = []
for pacient, num_pac in zip(pred_list_16,num_pa):
    pred_clas_mean = np.mean(pacient, axis = 0)
    list_pred_means_16.append(list(pred_clas_mean))
    print(num_pac, pred_clas_mean)
    

list_pred_means_16 = np.asarray(list_pred_means_16)
#%%Cell 19
#Define the true values for the confusion matrix.
true_values = [num_pa, ['Pancreatitis', 'Pancreatitis', 'Pancreatitis', 'Pancreatitis', 
                        'Normal', 'Normal', 'Pancreatitis', 'Pancreatitis', 
                         'Pancreatitis', 'Normal', 'Pancreatitis', 
                         'Pancreatitis', 'Normal', 'Pancreatitis', 
                         'Pancreatitis','Pancreatitis','Pancreatitis',
                        'Pancreatitis','Pancreatitis', 'Pancreatitis',
                        'Pancreatitis','Pancreatitis','Pancreatitis',
                        'Normal', 'Pancreatitis','Pancreatitis',
                        'Pancreatitis', 'Normal','Pancreatitis',
                        'Pancreatitis','Pancreatitis','Pancreatitis',
                        'Normal','Pancreatitis','Pancreatitis','Pancreatitis',
                        'Normal', 'Pancreatitis','Pancreatitis','Pancreatitis',
                        'Normal', 'Pancreatitis','Pancreatitis','Pancreatitis',
                        'Pancreatitis','Pancreatitis','Pancreatitis','Pancreatitis',
                        'Normal', 'Pancreatitis','Pancreatitis','Pancreatitis',
                        'Pancreatitis','Pancreatitis','Pancreatitis','Pancreatitis',
                        'Pancreatitis','Pancreatitis','Normal','Normal',
                        'Normal','Normal','Normal','Normal',
                        'Normal','Normal','Normal','Normal',
                        'Normal','Normal','Normal','Normal',
                        'Normal','Normal','Normal','Normal',
                        'Normal','Normal','Normal','Normal',
                        'Normal','Normal','Normal','Normal',]]

for i in range(len(true_values[0])):
    print(true_values[0][i], true_values[1][i])


true_values_encoded = [num_pa, encoder_classes(true_values[1], 2)]

for i in range(len(true_values_encoded[0])):
    print(true_values_encoded[0][i], true_values_encoded[1][i])


#%%Cell 19
#Call the function to make the confusion matrix and get the accuracy value.
true_values_vgg16 = true_values_encoded[1][:,1]
pred_values_vgg16 = np.round(list_pred_means_16[:,1])

confusion_matrix_func(true_values_vgg16, pred_values_vgg16, sorted(class_names), 'vgg16')
accuracy_matrix_func(true_values_vgg16, pred_values_vgg16, 'vgg16')

#%%Cell 19

#Train the VGG 19 model with the multiplication images and labels.
vgg19_model = mod_vgg19_model(Inputs)
vgg19_model.summary()


vgg19_model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

#%%Cell 20
#Train the VGG 19 model with the multiplication images and labels.
classification_vgg19 = vgg19_model.fit(x_train_mult_images, y_train_labels, validation_data=(x_test_mult_images,y_test_labels), batch_size = 8, epochs=epochs_vgg)

plot_loss_acc(classification_vgg19)

#%%Cell 21

#Predict the vgg19 model with the test multiplication images calculated above.
pred_list_19 = []
for pacient in pred_mult_images:
    preds_clas = vgg19_model.predict(np.asarray(pacient, np.float32))
    print(preds_clas)
    pred_list_19.append(preds_clas)
    
#Make the mean of all the patients
list_pred_means_19 = []    
for pacient, num_pac in zip(pred_list_19,num_pa):
    pred_clas_mean = np.mean(pacient, axis = 0)
    list_pred_means_19.append(list(pred_clas_mean))
    print(num_pac, pred_clas_mean)

list_pred_means_19 = np.asarray(list_pred_means_19)
#%%Cell 21

#Call the function to make the confusion matrix and get the accuracy value.
true_values_vgg19 = true_values_encoded[1][:,1]
pred_values_vgg19 = np.round(list_pred_means_19[:,1])

confusion_matrix_func(true_values_vgg19, pred_values_vgg19, sorted(class_names), 'vgg19')
accuracy_matrix_func(true_values_vgg19, pred_values_vgg19, 'vgg19')
    
        
#%%Cell 22



