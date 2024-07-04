import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense, Dropout, add
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, RMSprop, Adam
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu
import os
from preprocess import preprocess_image

captions_file = 'preprocessed_captions.csv'
image_features_file = 'image_features_pca.npy'

data = pd.read_csv(captions_file)
image_features = np.load(image_features_file)

captions = data['Opis'].values
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(captions)
max_length = max(len(seq) for seq in sequences)

def create_sequences(tokenizer, max_length, caption_list, feature):
    X1, X2, y = list(), list(), list()
    for caption in caption_list:
        seq = tokenizer.texts_to_sequences([caption])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

train_size = 0.7
val_size = 0.15
test_size = 0.15

image_paths = data['image'].values

X_train_image, X_test_image, y_train_caption, y_test_caption = train_test_split(image_paths, captions, test_size=(1-train_size), random_state=42)
X_val_image, X_test_image, y_val_caption, y_test_caption = train_test_split(X_test_image, y_test_caption, test_size=(test_size/(test_size + val_size)), random_state=42)

def create_data_splits(image_paths, captions):
    features = np.vstack([preprocess_image("data/Images/"+ path) for path in image_paths])
    return create_sequences(tokenizer, max_length, captions, features)

X1_train, X2_train, y_train = create_data_splits(X_train_image, y_train_caption)
X1_val, X2_val, y_val = create_data_splits(X_val_image, y_val_caption)
X1_test, X2_test, y_test = create_data_splits(X_test_image, y_test_caption)

def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(image_features.shape[1],))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

optimizers = {
    'sgd': SGD(learning_rate=0.001),
    'momentum': SGD(learning_rate=0.001, decay=0, momentum=0.9, nesterov=False),
    'rmsprop': RMSprop(learning_rate=0.001),
    'adam': Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
}

def evaluate_model(model, X1, X2, y_true, tokenizer, max_length):
    actual, predicted = list(), list()
    for i in range(len(X1)):
        yhat = generate_caption(model, tokenizer, X1[i], max_length)
        actual.append([y_true[i].split()])
        predicted.append(yhat.split())
    bleu = corpus_bleu(actual, predicted)
    return bleu

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

for name, optimizer in optimizers.items():
    model = define_model(vocab_size, max_length)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    
    filepath = f'model_{name}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    model.fit([X1_train, X2_train], y_train, epochs=20, verbose=2, validation_data=([X1_val, X2_val], y_val), callbacks=callbacks_list)
    
    bleu_score = evaluate_model(model, X1_test, X2_test, y_test_caption, tokenizer, max_length)
    print(f'Model {name} BLEU score: {bleu_score:.4f}')

print("Models have been trained, saved, and evaluated.")
