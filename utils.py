#%%Cell 1
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import random
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

from tensorflow import keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, smart_resize

from models import *

#Variables:

#%% Cell 2

#Create the function that opens the file and list the directions
def data_repetition(dataset_path, num_repeated_images):
    
    images_path_list = []
    masks_path_list = []
    
    
    for path, subdirs, files in os.walk(dataset_path):
        if 'Pancreas' in path:

            for image in files:
                if image.startswith('._'):
                    continue
                images_path_list.append(os.path.join(path, image))

        if 'Masks' in path:
            for mask in files:
                if mask.startswith('._'):
                    continue
                masks_path_list.append(os.path.join(path, mask))
                
    
    images_path_list = sorted(images_path_list)
    masks_path_list = sorted(masks_path_list)
    
    
    return images_path_list, masks_path_list

#%% Cell 3

#Prepare the images converting them into an array, resizing and removing noise.
def image_preparation(PIL_image, image_shape):
    img = img_to_array(PIL_image)
    img = cv2.medianBlur(img, 1)
    img = cv2.resize(img, image_shape) /255.0
    
    return img


#%% Cell 4

#Create the function that loads the images and split into images, masks, labels and multimplied images.
def data_preparation(path_images_list, path_masks_list, image_size, class_names_label):
    
    image_shape = (image_size, image_size)
    
    images = []
    masks = []
    labels = []
    mult_images = []
    
    for image,mask in zip(path_images_list, path_masks_list):
   
        for item in class_names_label:
            if item in image:
                label = item
        try:
            original_image = load_img(image, color_mode='grayscale')
            original_mask = load_img(mask, color_mode='grayscale')
        except PIL.UnidentifiedImageError:
            print(image)
            print(mask)
        
        img = image_preparation(original_image, image_shape)
        images.append(img)
        labels.append([label])
        mask = image_preparation(original_mask, image_shape)
        masks.append(mask)
        
        mult_image = np.zeros(image_shape)
        for row in range(image_size):
            for col in range(image_size):
                mult_image[row][col] = img[row][col]*mask[row][col]
                
        mult_images.append(mult_image)
    
    images = np.asarray(images,dtype=np.float32)
    print('Images:',images.shape)
    labels = np.asarray(labels)
    print('Labels:',labels.shape)
    
    masks = np.asarray(masks,dtype=np.int32)
    print('Masks:',masks.shape)
    
    mult_images = np.asarray(mult_images, dtype=np.float32)
    print('Multiplied images:',mult_images.shape)
    
    
    return images, labels, masks, mult_images

#%% Cell 5
#This function shows the images and the superposed one.
def show_images(img1, img2, img3, img_name):
    
    alpha = 0.5
    
    if img1.dtype == img2.dtype:
        
        if img3 is None:
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,20))
        
            ax1.imshow(img1, cmap = 'gray')
    
            ax2.imshow(img2, cmap = 'gray')
            
            fusion = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
            
            ax3.imshow(fusion)
    
            plt.show()
            
            if img_name not in (0, None):
                plt.savefig(img_name)
   
        else:
            fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(20,20))
            
            ax1.imshow(img1, cmap = 'gray')
    
            ax2.imshow(img2, cmap = 'gray')
            
            fusion = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
            
            ax3.imshow(fusion)
            
            ax4.imshow(img3, cmap = 'gray')
    
            plt.show()
            
            if img_name not in (0, None):
                
                fig.savefig(img_name)
                
    else:
        print("They do not have the same type, try to convert one of them.")
    
        

#%% Cell 6
#This function generates and index to split data into train and test
def get_index_split(len_data):
    # length of data as indices of n_data
    n_data = len_data
    
    test_set = 0.8

    # get n_samples based on percentage of n_data
    n_samples = int(n_data * test_set)

    # make n_data a list from 0 to n
    n_data  = list(range(n_data))

    # randomly select from range of n_data as indices
    np.random.seed(42)
    idx_train = np.random.choice(n_data, n_samples, replace=False)
    idx_test = list(set(n_data) - set(idx_train))
    
    return idx_train, idx_test



#%% Cell 7

#This function encodes the categorical labels
def encoder_classes(labels, num_class):
    Lab_Enc = preprocessing.LabelEncoder()
    labels = Lab_Enc.fit_transform(labels)
    labels = keras.utils.to_categorical(labels, num_classes=num_class)
    labels = np.asarray(labels,dtype=np.int32)
    
    return labels
    
#This function splits the data into train and test
def split_data(images, labels, index_list, masks, mult_images):
    
    
    x_images = []
    x_mult_images = []
    y_labels = []
    y_masks = []

    for index in index_list:
        x_images.append(images[index])
        x_mult_images.append(mult_images[index])
        y_labels.append(labels[index])
        y_masks.append(masks[index])

    x_images = np.asarray(x_images,dtype=np.float32)
    print('x_train_images', x_images.shape)
    
    x_mult_images = np.asarray(x_mult_images,dtype=np.float32)
    print('x_train__mult_images', x_mult_images.shape)
    
    num_clas = 2
    y_labels = encoder_classes(y_labels, num_clas)
    print('y_train_labels', y_labels.shape)
    
    y_masks = np.asarray(y_masks,dtype=np.int32)
    print('y_train_masks', y_masks.shape)
        
    return x_images, x_mult_images, y_labels, y_masks


    
#%% Cell 8
#This function splits the whole data by images-masks and multiplied images- labels
def split_train_test(Images, Labels, Masks, mult_images):
    
    idx_train, idx_test = get_index_split(len(Images))
    
    x_train_images, x_train_mult_images, y_train_labels, y_train_masks = split_data(Images, Labels, idx_train, Masks, mult_images)
    x_test_images, x_test_mult_images, y_test_labels, y_test_masks = split_data(Images, Labels, idx_test, Masks, mult_images )
        
    return x_train_images, x_train_mult_images, y_train_labels, y_train_masks, x_test_images,x_test_mult_images, y_test_labels, y_test_masks
        
    
    
#%% Cell 9

#Shows and round the predictions of the segmentation model
def show_preds(pacient, preds, pred_path, test_path_list):
    
    z = 0
    multiplied_images = []
    
        
    for i in range(len(pacient)):

        x = len(os.listdir(pred_path))
    
    
        img_name = os.path.join(pred_path, 'prediction_' +test_path_list[i][-12:-4]+'_'+str(x)+'.jpg')
        print(img_name)
        
        
        pred = preds[i]
        empty_pred = np.zeros((len(pred), len(pred)))
        print(z)
        
        for x in range(len(pred)):
            for y in range(len(pred)):
                if pred[x][y] < 0.5:
                    empty_pred[x][y] = 0
                else:
                    empty_pred[x][y] = 1
        
        pred = np.squeeze(empty_pred).astype(np.float32)
        test = np.squeeze(pacient[i]).astype(np.float32)
        
        empty_image = np.zeros((len(pred), len(pred)))
        for row in range(len(pred)):
            for col in range(len(pred)):
                empty_image[row][col] = pred[row][col] * test[row][col]
        
        
        print(z)
        z +=1
        multiplied_images.append(empty_image)
    


        show_images(pred, test, empty_image, img_name)
        plt.show()
        
    return multiplied_images

#%% Cell 10

#This function put all masks together to get the shape of the whole pancreas
def total_pancreas(list_preds):
    image_size = list_preds[0].shape[0]
    complete_pancreas = np.zeros((image_size, image_size))
    
    for pancreas in list_preds:
        for i in range(image_size):
            for j in range(image_size):
                a = complete_pancreas[i][j]
                b = pancreas[i][j]
                complete_pancreas[i][j] = np.maximum(a, b)
                if complete_pancreas[i][j] < 0.5:
                    complete_pancreas[i][j] = 0
                else:
                    complete_pancreas[i][j] = 1
                
    crop_pancreas = complete_pancreas[50:150, 75:225]
    image_shape = 256
    crop_pancreas = cv2.resize(crop_pancreas, (image_size, image_size))
                
    show_images(complete_pancreas, crop_pancreas, None, None)
    
    return complete_pancreas


#%% Cell 11

#Calculates the area ratio depending on the shape of the image
def area_pancreas(pancreas, image_size):
    
    total_area = image_size*image_size
    areas_list = []

    pixels = 0
    pancreas_area = 0
    for row in pancreas:
        for c in row:     
            if c == 1:
                pixels += 1
                    
    print('Number pf pixels: ', pixels)
    print('Total area: ', total_area)
    pancreas_area = pixels/total_area*100
    print('Pancreas area Ratio: ', round(pancreas_area, 2),'%')
    areas_list.append(pancreas_area)

#%% Cell 12

"""def repeated_classification(labels_images, num_repeated_images):
    
    new_data_classification = []
    
    classes = []
    images = []
    
    print(type(classes))
    np.random.seed(42)
    
    for i in range(num_repeated_images):
        number = random.randint(0, len(labels_images[1])-1)  #Pick a number to select an image & mask
        classes.append([labels_images[1][number]])
        
        print(len(labels_images[0][number]))
        
        images.append(labels_images[0][number])
    
    classes = np.asarray(classes)
    
    images = np.asarray(images, dtype=np.float32)
        
    return classes, images"""


#%% Cell 13

#Creates the confusion matrix depending on the true and predicted values
def confusion_matrix_func(true_values, pred_values,labels, model_name):
    
    conf_matrix = confusion_matrix(true_values,pred_values)
    
    print('The confusion matrix from Model: '+model_name)
    print(conf_matrix)
    
    ax = sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix with labels from Model: ' + model_name + '\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    
    plt.show()
    
#%% Cell 14

#Creates the accuracte value depending on the true and predicted values
def accuracy_matrix_func(true_values, pred_values, model_name):
    
    acc_value = accuracy_score(true_values,pred_values)
    
    print('Accuracy value with labels from Model: '+model_name)
    print(str(np.round(acc_value*100,2)) +'%')