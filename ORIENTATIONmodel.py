# -*- coding: utf-8 -*-
"""
Created on Sun May 28 23:44:42 2023

@author: joao_
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
import matplotlib.pyplot as plt


data_dir = 'faces/an2i'

# Define the target size for resizing the images
target_size = (64, 64)

images = []
labels = []


#%%
for filename in os.listdir(data_dir):
    if filename.endswith('.pgm'):
        
        # Extract the label from the file name
        label = filename.split('_')[1].split('.')[0]

        
        # Load the image
        img = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_GRAYSCALE)
        
        # Resize the image while maintaining aspect ratio
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        # Append the image and label to the lists
        images.append(img_resized)
        labels.append(label)
        

# Convert the lists to numpy arrays
images = np.array(images)
labels = np.array(labels)



# Normalize the images
images = images / 255.0

# Convert labels to one-hot encoding
label_map = {'left': 0, 'right': 1, 'straight': 2, 'up': 3}
labels_encoded = np.array([label_map[label] for label in labels])
labels_onehot = tf.keras.utils.to_categorical(labels_encoded)






# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_onehot, test_size=0.2, random_state=42)


model = tf.keras.Sequential([
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
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=6, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print('Test Loss for orientation of the head:', loss)
print('Test Accuracy for orientation of the head:', accuracy)


x_val=[]
for i in range(6):
  x_val.append(i+1)
# Plot history: Categorical crossentropy & Accuracy
plt.subplot(2,1,1)
plt.plot(x_val,history.history['loss'], label='Categorical crossentropy (training data)')
plt.plot(x_val,history.history['val_loss'], label='Categorical crossentropy (validation data)')
plt.title('Model results for orientation of faces classification')
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
#%%

imagem = Image.open("NewImage/straight_neutral_open.pgm")
imagem_array = np.array(imagem)
plt.imshow(imagem_array, cmap='gray')
plt.axis('off')
plt.show()



new_image = cv2.imread('NewImage/straight_neutral_open.pgm', 0)
resized_new_image = cv2.resize(new_image, (64, 64))
input_image = np.expand_dims(resized_new_image, axis=-1)
input_image = np.expand_dims(input_image, axis=0)
input_image = input_image.astype('float32') / 255.0

prediction = model.predict(input_image)

reverse_label_map = {v: k for k, v in label_map.items()}  # Create reverse mapping


# Convert the prediction probabilities to labels
orientation_prediction = reverse_label_map[np.argmax(prediction[0])]

print(f'Orientation: {orientation_prediction}')

imagem = Image.open("NewImage/up_neutral_open.pgm")
imagem_array = np.array(imagem)
plt.imshow(imagem_array, cmap='gray')
plt.axis('off')
plt.show()

new_image = cv2.imread('NewImage/up_neutral_open.pgm', 0)
resized_new_image = cv2.resize(new_image, (64, 64))
input_image = np.expand_dims(resized_new_image, axis=-1)
input_image = np.expand_dims(input_image, axis=0)
input_image = input_image.astype('float32') / 255.0

prediction = model.predict(input_image)

reverse_label_map = {v: k for k, v in label_map.items()}  # Create reverse mapping


# Convert the prediction probabilities to labels
orientation_prediction = reverse_label_map[np.argmax(prediction[0])]

print(f'Orientation: {orientation_prediction}')

imagem = Image.open("NewImage/left_neutral_open.pgm")
imagem_array = np.array(imagem)
plt.imshow(imagem_array, cmap='gray')
plt.axis('off')
plt.show()



new_image = cv2.imread('NewImage/left_neutral_open.pgm', 0)
resized_new_image = cv2.resize(new_image, (64, 64))
input_image = np.expand_dims(resized_new_image, axis=-1)
input_image = np.expand_dims(input_image, axis=0)
input_image = input_image.astype('float32') / 255.0

prediction = model.predict(input_image)

reverse_label_map = {v: k for k, v in label_map.items()}  # Create reverse mapping


# Convert the prediction probabilities to labels
orientation_prediction = reverse_label_map[np.argmax(prediction[0])]

print(f'Orientation: {orientation_prediction}')







