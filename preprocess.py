import pandas as pd
import re
import numpy as np
import tensorflow as tf
from keras import models
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

custom_stop_words = set("""
а ако али да јер јесте онај она оно оног оном оним оне оној ону који која које тај та то тог том томе тим те тој ту такав таква такво таквог таквом таквим такве таквој такву онакав онаква онакво онаквог онаквом онаквим онакве онаквој онакву толики толика толико толиког толиком толиким толике толикој толику онолики онолика онолико оноликог оноликом оноликим онолике оноликој онолику овај ова ово овог овом овим ове овој ову овакав оваква овакво оваквог оваквом оваквим овакве оваквој оволики оволика оволико оволиког оволиком оволиким оволике оволикој оволику бити јесте јесам сам си смо сте јеси јесмо jесте као ја неки нека неко неког неком неким неке некoj један једна једно једног једном једним једне једној једну неки нека неко неког неком неким неке некој један једна једно једног једном једним једне једној једну о скоро готово безмало сада одмах баш управо спреман био била било ко који која које шта какав каква какво каквог каквом каквим какве каквој какву колики колика колико колике где куда одакле откуда када кад чим зашто што како који која које којег кога којем којим које којој коју којом такође још штавише то оно код при или
""".split())

print(custom_stop_words)

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in custom_stop_words]
    return 'startseq ' + ' '.join(words) + ' endseq'

captions_file = 'captions_serbian.txt'
data = pd.read_csv(captions_file)
data['Opis'] = data['Opis'].apply(preprocess_text)
print(data.head())

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