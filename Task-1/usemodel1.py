import tensorflow as tf
from keras_preprocessing.image import load_img
import pandas as pd
import os
import numpy as np
from tqdm.notebook import tqdm

def prepareImages(dir):
    image_paths = []
    image_names = []
    for imagename in os.listdir(os.path.join(dir)):
        image_paths.append(os.path.join(dir, imagename))
        image_names.append(imagename)
    return image_paths, image_names

def extract_features(images):
    features = []
    for image in tqdm(images):
        try:
            img = load_img(image, target_size=(236, 236))
            img = np.array(img)
            features.append(img)
        except:
            print("skipping img", image)
    features = np.array(features)
    features = features.reshape(features.shape[0], 236, 236, 3)  # Reshape all images in one go
    return features


TEST_DIR="Task-1/Data/Test"

test = pd.DataFrame()
test["imagepath"], test["Id"] = prepareImages(TEST_DIR)
test_features=extract_features(test["imagepath"])
x_test = test_features/255.0

new_model = tf.keras.models.load_model('Task-1/model.keras')
test["result"] = new_model.predict(x=x_test).argmax(axis=1)
test["Label"] = test["result"].map({1: "Real", 0: "AI"})

result = pd.DataFrame()
result["Id"], result["Label"] = test["Id"] , test["Label"]
result["Id"]= result["Id"].str.replace(".jpg", "", regex=False)
result.to_csv("Task-1/output/result1.csv", index= False)

