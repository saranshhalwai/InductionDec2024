from keras.utils import to_categorical
from keras_preprocessing.image import load_img, ImageDataGenerator
from keras.models import Sequential
from keras.applications import MobileNetV2, ResNet152, VGG16, EfficientNetB0, InceptionV3
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels

def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, target_size=(236, 236))
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(features.shape[0], 236, 236, 3)  # Reshape all images in one go
    return features

TRAIN_DIR = "Task-1/Data/Train"

train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)

datagen = ImageDataGenerator( #to prepare dataset augmentation
    rescale=1./255,                  
    rotation_range=20,               
    width_shift_range=0.2,           
    height_shift_range=0.2,          
    shear_range=0.2,                 
    zoom_range=0.2,                  
    horizontal_flip=True,            
    fill_mode='nearest'              
)

train_features = extract_features(train['image'])

x_train = train_features / 255.0

le = LabelEncoder()
le.fit(train['label'])
y_train = le.transform(train['label'])
y_train = to_categorical(y_train, num_classes=2)

# Convert x_train and y_train to lists for augmentation
x_train = list(x_train)
y_train = list(y_train)

# Augment the dataset
augmented_images = []
augmented_labels = []

for img, label in zip(x_train, y_train):
    img = img.reshape((1,) + img.shape)  # Add batch dimension for datagen.flow
    i = 0
    for aug_img in datagen.flow(img, batch_size=1):
        augmented_images.append(aug_img[0])  # Remove batch dimension
        augmented_labels.append(label)      # Add the corresponding label
        i += 1
        if i >= 3:  # Generate 5 augmented images per input image
            break

# Combine original and augmented data
x_train_augmented = np.array(x_train + augmented_images)
y_train_augmented = np.array(y_train + augmented_labels)

print("Original dataset size:", len(x_train))
print("Augmented dataset size:", len(x_train_augmented))

model = Sequential()
# Convolutional layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(236, 236, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train_augmented, y=y_train_augmented, batch_size=25, epochs=20, validation_split=0.1)

model.save("Task-1/model.keras")