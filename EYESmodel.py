# -*- coding: utf-8 -*-
"""
Created on Sun May 28 23:17:41 2023

@author: joao_
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


from PIL import Image
import matplotlib.pyplot as plt
#%%
data_dir = 'faces/an2i'

images = []
labels = []

for filename in os.listdir(data_dir):
    if filename.endswith('.pgm'):
        file_components = os.path.splitext(filename)[0].split('_')
        
        
        if len(file_components) == 4:
            name, orientation, expression, eyes = file_components
            
            # Load the image using OpenCV
            img = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_GRAYSCALE)
            
            # Resize the image to a desired size (e.g., 64x64)
            img = cv2.resize(img, (64, 64))
            
            # Normalize the pixel values to the range [0, 1]
            img = img.astype('float32') / 255.0
            
            images.append(img)
            labels.append(eyes)
            
        elif len(file_components) == 5:     
             name, orientation, expression, eyes, size = file_components
             
             # Load the image using OpenCV
             img = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_GRAYSCALE)
             
             # Resize the image to a desired size (e.g., 64x64)
             img = cv2.resize(img, (64, 64))
             
             # Normalize the pixel values to the range [0, 1]
             img = img.astype('float32') / 255.0
             
             images.append(img)
             labels.append(eyes)
   
             
             
images = np.array(images)
labels = np.array(labels)


# Perform one-hot encoding on the labels
label_mapping = {'open': 0, 'sunglasses': 1}
encoded_labels = np.array([label_mapping[label] for label in labels])

X_train, X_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)





# model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Reshape the input data to match the model's input shape
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Train the model
history = model.fit(X_train, y_train, epochs=7, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print('Test Loss for the Eyes:', loss)
print('Test Accuracy for the Eyes:', accuracy)

x_val=[]
for i in range(7):
  x_val.append(i+1)
# Plot history: Categorical crossentropy & Accuracy
plt.subplot(2,1,1)
plt.plot(x_val,history.history['loss'], label='Categorical crossentropy (training data)')
plt.plot(x_val,history.history['val_loss'], label='Categorical crossentropy (validation data)')
plt.title('Model results for  the eyes of faces classification')
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

reverse_label_map = {v: k for k, v in label_mapping.items()}  # Create reverse mapping


# Convert the prediction probabilities to labels
eyes_prediction = reverse_label_map[np.argmax(prediction[0])]

print(f'Eyes: {eyes_prediction}')

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

reverse_label_map = {v: k for k, v in label_mapping.items()}  # Create reverse mapping


# Convert the prediction probabilities to labels
eyes_prediction = reverse_label_map[np.argmax(prediction[0])]

print(f'Eyes: {eyes_prediction}')

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

reverse_label_map = {v: k for k, v in label_mapping.items()}  # Create reverse mapping


# Convert the prediction probabilities to labels
eyes_prediction = reverse_label_map[np.argmax(prediction[0])]

print(f'Eyes: {eyes_prediction}')
