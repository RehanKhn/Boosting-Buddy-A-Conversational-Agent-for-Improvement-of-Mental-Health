from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer

class IntentClassifier:
    def __init__(self):
        pass

    def load_file(self, filename):
        return pd.read_csv(filename, encoding="latin1",
                         names=["Sentence", "Intent"]) 

    def get_intent(self):
        return self.load_file("Dataset.csv")["Intent"]
    
    def get_unique_intent(self):
        return list(set(self.get_intent()))

    def get_sentences(self):
        return list(self.load_file("Dataset.csv")["Sentence"])

    def load_dataset(self, filename):
        return (self.get_intent(), self.get_unique_intent(), self.get_sentences())

    def cleaning(self, sentences):
        words = []
        for s in self.get_sentences():
            clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
            w = word_tokenize(clean)
            words.append([i.lower() for i in w])
        return words

    def create_tokenizer(self, words, filters):
        token = Tokenizer(filters=filters)
        token.fit_on_texts(words)
        return token

    def max_length(self, words):
        return (len(max(words, key=len)))

    def get_max_length(self):
        return self.max_length(self.get_cleaned_words())

    def get_cleaned_words(self):
        return self.cleaning(self.get_sentences())

    def get_word_tokenizer(self):
        return self.create_tokenizer(self.get_cleaned_words(), '!"#$%&()*+,-./:;<=>?@[]\\^_`{|}~')

    def get_vocab_size(self):
        return len(self.get_word_tokenizer().word_index) + 1

    def encoding_doc(self, token, words):
        return(token.texts_to_sequences(words))

    def get_encoded_doc(self):
        return self.encoding_doc(self.get_word_tokenizer(), self.get_cleaned_words())

    def padding_doc(self, encoded_doc, max_length):
        return(pad_sequences(self.get_encoded_doc(), maxlen=self.get_max_length(), padding="post"))

    def get_padded_doc(self):
        return self.padding_doc(self.get_encoded_doc(), self.get_max_length())

    def get_output_tokenizer(self):
        return self.create_tokenizer(self.get_unique_intent(), filters='!"#$%&()*+,-/:;<=>?@[\]^`{|}~')

    def get_encoded_output(self):
        encoded_output = self.encoding_doc(
            self.get_output_tokenizer(), self.get_intent())
        encoded_output = np.array(encoded_output).reshape(
            len(encoded_output), 1)
        return encoded_output

    def one_hot(self, encode):
        o = OneHotEncoder(sparse=False)
        return(o.fit_transform(encode))

    def output_one_hot(self):
        return self.one_hot(self.get_encoded_output())

    def create_model(self, vocab_size, max_length):
        model = Sequential()
        model.add(Embedding(self.get_vocab_size(), 128,
                            input_length=self.get_max_length(), trainable=False))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(21, activation="softmax"))
        return model

    def train_model(self):
        train_X, val_X, train_Y, val_Y = train_test_split(
            self.get_padded_doc(), self.output_one_hot(), shuffle=True, test_size=0.2)
        model = self.create_model(self.get_vocab_size(), self.get_max_length())
        model.compile(loss="categorical_crossentropy",
                      optimizer="adam", metrics=["accuracy"])
        filename = 'model.h5'
        checkpoint = ModelCheckpoint(
            filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        hist = model.fit(train_X, train_Y, epochs=50, batch_size=32, validation_data=(val_X, val_Y), callbacks=[checkpoint]
                         )

    def load_model(self):
        return load_model("model.h5")

    def predictions(self, text):
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
        test_word = word_tokenize(clean)
        test_word = [w.lower() for w in test_word]
        test_ls = self.get_word_tokenizer().texts_to_sequences(test_word)

        if [] in test_ls:
            test_ls = list(filter(None, test_ls))

        test_ls = np.array(test_ls).reshape(1, len(test_ls))

        x = self.padding_doc(test_ls, self.get_max_length())

        pred = self.load_model().predict(x)

        return pred

    def get_final_output(self, pred, classes):
        predictions = pred[0]

        classes = np.array(classes)
        ids = np.argsort(-predictions)
        classes = classes[ids]
        predictions = -np.sort(-predictions)
        outputPred = predictions[1]
        for i in range(pred.shape[1]):
            if(outputPred <= predictions[i]):
                outputPred = predictions[i]
        outputClass = classes[np.where(predictions == outputPred)[0][0]]
        return outputClass

    def get_class(self, text):
        pred = self.predictions(text)
        return self.get_final_output(pred, self.get_unique_intent())
         
