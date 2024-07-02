import pandas as pd
import re
import numpy as np
import tensorflow as tf
from keras import models
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  
    text = text.lower() 
    text = text.replace('"', "")
    return  'startseq ' + text + ' endseq'

captions_file = 'captions_serbian.txt'
data = pd.read_csv(captions_file)
data['Opis'] = data['Opis'].apply(preprocess_text)

def load_vgg16():
    modelvgg = VGG16(include_top=True, weights=None)
    modelvgg.load_weights("output/vgg16_weights_tf_dim_ordering_tf_kernels.h5")
    modelvgg.layers.pop()
    modelvgg = models.Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-1].output)
    modelvgg.summary()
    return modelvgg

vgg16_model = load_vgg16()

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

image_paths = data['image'].values
preprocessed_images = np.vstack([preprocess_image("data/Images/"+ path) for path in image_paths])
features = vgg16_model.predict(preprocessed_images)
print("Feature vectors shape:", features.shape) 

pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)

data.to_csv('preprocessed_captions.csv', index=False)

np.save('image_features_pca.npy', pca_result)

plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.title('PCA of VGG16 Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

print(f'Successfully processed and saved {len(data)} captions and image features.')