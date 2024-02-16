# -*- coding: utf-8 -*-
"""
Created on Sat May 27 15:34:27 2023

@author: joao_
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt




#%%
data_dir = 'faces/an2i'

#  target size for resizing the images
target_size = (64, 64)

images = []
labels_expression = []


for filename in os.listdir(data_dir):
    if filename.endswith('.pgm'):
        
        # Extract the label from the file name
        label_expression = filename.split('_')[2].split('.')[0]
        
        # Load the image
        img = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_GRAYSCALE)
        
        # Resize the image while maintaining aspect ratio
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        images.append(img_resized)
        labels_expression.append(label_expression)

# Convert the lists to numpy arrays
images = np.array(images)
labels_expression = np.array(labels_expression)


# Normalize the images
images = images / 255.0


# Convert labels to one-hot encoding


label_map_expression = {'angry': 0, 'happy': 1, 'neutral': 2, 'sad': 3}
labels_encoded_expression = np.array([label_map_expression[label_expression] for label_expression in labels_expression])
labels_onehot_expression = tf.keras.utils.to_categorical(labels_encoded_expression)




# Split the data into training and testing sets

X_train_expression, X_test_expression, y_train_expression, y_test_expression = train_test_split(images, labels_onehot_expression, test_size=0.2, random_state=42)


batch_size = 100
nb_epoch = 50


# Build the CNN model
model1 = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=target_size + (1,)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])

# Compile and train the model
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



history = model1.fit(X_train_expression, y_train_expression, batch_size=batch_size, epochs=nb_epoch,
          verbose=1, validation_data=(X_test_expression, y_test_expression))
score = model1.evaluate(X_test_expression, y_test_expression, verbose=0)
print('Test loss for expression:', score[0])
print('Test accuracy for expression:', score[1])



x_val=[]
for i in range(nb_epoch):
  x_val.append(i+1)
# Plot history: Categorical crossentropy & Accuracy
plt.subplot(2,1,1)
plt.plot(x_val,history.history['loss'], label='Categorical crossentropy (training data)')
plt.plot(x_val,history.history['val_loss'], label='Categorical crossentropy (validation data)')
plt.title('Model results for faces expression')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper right")
plt.subplot(2,1,2)
plt.plot(x_val,history.history['accuracy'], label='Accuracy (training data)')
plt.plot(x_val,history.history['val_accuracy'], label='Accuracy (validation data)')
plt.ylabel('Accuracy value')
plt.xlabel('No. epoch')
plt.legend(loc="lower right")
plt.show()


#%% CNN with dropout

#parameter to train the model
batch_size = 128
nb_epoch = 50
nb_filters=32
kernel_size=(3,3)
pool_size=(2,2)
input_shape=(32,32,3)



model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=target_size + (1,)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout((0.3)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout((0.3)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dropout((0.3)),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])



# Compile and train the new model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])




#Training the model
history = model.fit(X_train_expression, y_train_expression, batch_size=batch_size, epochs=nb_epoch,
          verbose=1, validation_data=(X_test_expression, y_test_expression))
score = model.evaluate(X_test_expression, y_test_expression, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


x_val=[]
for i in range(nb_epoch):
  x_val.append(i+1)
# Plot history: Categorical crossentropy & Accuracy
plt.subplot(2,1,1)
plt.plot(x_val,history.history['loss'], label='Categorical crossentropy (training data)')
plt.plot(x_val,history.history['val_loss'], label='Categorical crossentropy (validation data)')
plt.title('Model results for faces expression (with dropout)')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper right")
plt.subplot(2,1,2)
plt.plot(x_val,history.history['accuracy'], label='Accuracy (training data)')
plt.plot(x_val,history.history['val_accuracy'], label='Accuracy (validation data)')
plt.ylabel('Accuracy value')
plt.xlabel('No. epoch')
plt.legend(loc="lower right")
plt.show()
