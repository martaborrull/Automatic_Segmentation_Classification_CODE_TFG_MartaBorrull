# Automatic_Segmentation_Classification_CODE_TFG_MartaBorrull

My name is Marta Borrull and this is my bachelor's thesis: Automatic Segmentation and Classfication for Pancreatitis.

Acute Pancreatitis is the inflammation of the pancreas. This happens when digestive juices or enzymes attack the pancreas. 
In Spain, this disease is range with 49,3 cases per 100 000 population and this involves to increase the number of hospitalize patients because of this pathology.
For this reason, using technologies that are on the rise nowadays as Deep Learning and Neural Networks, we implemented some tools that can help medical staff to be less time-consume and increase their efficiency when they diagnose a patient. With the collaboration of the Vall d’Hebron University Hospital, that has provided the MRIs from anonymised patients, it has been realized a segmentation model that detects and resects the pancreas, and a classifier that predict if a patient has this disease by using images and medical variables or labels. These networks can help to monitor and evaluate the evolution of the patients with pancreatitis.


There are 3 documents needed to download to see execute the code:
- models.py: There are the models used for segmentation (U-net) and for classification (VGG16 and VGG19)
- functions.py: There are multiple functions that are called in the main file such as load the data, split into train and test, etc.
- Main.py: The main file that will be executed.

In order to execute this code:
- There are some parameters that you can change depending on the size of the image, if they are in RGB mode, etc.
- You will need to indicate the location of the images and masks and also the test images. 
