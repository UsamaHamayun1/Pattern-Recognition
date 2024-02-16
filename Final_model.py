# -*- coding: utf-8 -*-
"""
Created on Mon May 29 01:39:28 2023

@author: joao_
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical


import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier








def create_model(kernel_size=(3, 3), pool_size=(2, 2), dropout=0.3):

    
    model1 = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=target_size + (1,)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout((0.3)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout((0.3)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dropout((0.4)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='softmax')
    ])


    model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model1


#%%
directory = 'faces/an2i'



images = []

labels=[]



for filename in os.listdir(directory):
    if filename.endswith('.pgm'):  # Only process .pgm files
        # Remove the file extension
        file_name_without_extension = os.path.splitext(filename)[0]
        
        # Split the file name using underscores
        parts = file_name_without_extension.split('_')
        
        # Extract the desired parts based on their positions
        extracted_parts = '_'.join(parts[1:4])
        
        labels.append(extracted_parts)
        
        img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
        # Resize the image to a desired size 
        img = cv2.resize(img, (64, 64))
        # Normalize the pixel values to the range [0, 1]
        img = img.astype('float32') / 255.0
        
        # Append the image and corresponding label to the lists
        images.append(img)
        
        
        
        
        
        
# Convert the lists to numpy arrays
images = np.array(images)
labels = np.array(labels)


# Perform one-hot encoding on the labels
label_mapping = {'left_angry_open': 0, 'left_angry_sunglasses': 1,'left_happy_open': 2,
                 'left_happy_sunglasses': 3,'left_neutral_open': 4,'left_neutral_sunglasses': 5,
                 'left_sad_open': 6,'left_sad_sunglasses': 7,'right_angry_open': 8,
                 'right_angry_sunglasses': 9,'right_happy_open': 10,'right_happy_sunglasses': 11,
                 'right_neutral_open': 12,'right_neutral_sunglasses': 13,'right_sad_open': 14,
                 'right_sad_sunglasses': 15,'straight_angry_open': 16,'straight_angry_sunglasses': 17,
                 'straight_happy_open': 18,'straight_happy_sunglasses': 19,'straight_neutral_open': 20,
                 'straight_neutral_sunglasses': 21,'straight_sad_open': 22,'straight_sad_sunglasses': 23,
                 'up_angry_open': 24,'up_angry_sunglasses': 25,'up_happy_open': 26,
                 'up_happy_sunglasses': 27,'up_neutral_open': 28,'up_neutral_sunglasses': 29,
                 'up_sad_open': 30,'up_sad_sunglasses': 31}
encoded_labels = np.array([label_mapping[label] for label in labels])
encoded_labels = to_categorical(encoded_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

batch_size = 100
nb_epoch = 50
target_size = (64, 64)


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



history = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
          verbose=1, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss for expression:', score[0])
print('Test accuracy for expression:', score[1])

x_val=[]
for i in range(nb_epoch):
  x_val.append(i+1)
# Plot history: Categorical crossentropy & Accuracy
plt.subplot(2,1,1)
plt.plot(x_val,history.history['loss'], label='Categorical crossentropy (training data)')
plt.plot(x_val,history.history['val_loss'], label='Categorical crossentropy (validation data)')
plt.title(' First Final Model results for faces classification')
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



model1 = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=target_size + (1,)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout((0.3)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout((0.3)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dropout((0.4)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='softmax')
])



# Compile the new model
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])




#Training the model
history = model1.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
          verbose=1, validation_data=(X_test, y_test))
score = model1.evaluate(X_test, y_test, verbose=0)
print('Test loss (with dropout):', score[0])
print('Test accuracy (with dropout):', score[1])


x_val=[]
for i in range(nb_epoch):
  x_val.append(i+1)
# Plot history: Categorical crossentropy & Accuracy
plt.subplot(2,1,1)
plt.plot(x_val,history.history['loss'], label='Categorical crossentropy (training data)')
plt.plot(x_val,history.history['val_loss'], label='Categorical crossentropy (validation data)')
plt.title(' Second Final Model results for faces classification (with dropout)')
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

#%% Defining which are the best values for the parameters


model = KerasClassifier(build_fn=create_model)

# Define the parameter grid for the grid search
param_grid = {
    'batch_size': [100, 128],
    'nb_epoch': [40, 50],
    'kernel_size': [(3, 3), (5, 5)],
    'pool_size': [(2, 2), (3, 3)],
    'dropout': [0, 0.3, 0.5]   
}

# Create the GridSearchCV object
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)

# Perform the grid search
grid_result = grid.fit(images, encoded_labels)

# Print the best parameters and score
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))



#%%  BEST model



#parameter to train the model
batch_size = 100
nb_epoch = 40
kernel_size=(5,5)
pool_size=(2,2)
input_shape=(32,32,3)
dropout = 0



model1 = tf.keras.Sequential([
    layers.Conv2D(32, kernel_size, activation='relu', input_shape=target_size + (1,)),
    layers.MaxPooling2D(pool_size),
    layers.Dropout((dropout)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout((dropout)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dropout((dropout)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='softmax')
])



# Compile the new model
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])




#Training the model
history = model1.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
          verbose=1, validation_data=(X_test, y_test))
score = model1.evaluate(X_test, y_test, verbose=0)
print('Test loss of best model:', score[0])
print('Test accuracy of best model:', score[1])


x_val=[]
for i in range(nb_epoch):
  x_val.append(i+1)
# Plot history: Categorical crossentropy & Accuracy
plt.subplot(2,1,1)
plt.plot(x_val,history.history['loss'], label='Categorical crossentropy (training data)')
plt.plot(x_val,history.history['val_loss'], label='Categorical crossentropy (validation data)')
plt.title('Final Model (with best parameters) results for faces classification')
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

