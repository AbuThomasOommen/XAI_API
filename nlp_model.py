import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.pipeline import TransformerMixin
from sklearn.base import BaseEstimator
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import joblib
        
# Load Data
df = pd.read_csv(r"C:\Users\abuth\OneDrive\Documents\Thesis-Project\tweets.csv", header=None)
df = df[[1,7]]
df.columns = ['tweet','label']

def clean_text(s):
    s = re.sub(r'http\S+', '', s)
    s = re.sub('(RT|via)((?:\\b\\W*@\\w+)+)', ' ', s)
    s = re.sub(r'@\S+', '', s)
    s = re.sub('&amp', ' ', s)
    return s
df['clean_tweet'] = df['tweet'].apply(clean_text)

# Encode Categorical Variable
X = df['clean_tweet']
encode_cat = {"label":     {"['neutral']": 0, "['positive']": 1, "['negative']": 2},
             }
y_df = df.replace(encode_cat)
y = y_df['label']

# Split Train Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=1221)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

vocab_size = 20000  # Max number of different word, i.e. model input dimension
maxlen = 80  # Max number of words kept at the end of each text

class TextsToSequences(Tokenizer, BaseEstimator, TransformerMixin):
    """ Sklearn transformer to convert texts to indices list 
    (e.g. [["the cute cat"], ["the dog"]] -> [[1, 2, 3], [1, 4]])"""
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        
    def fit(self, texts, y=None):
        self.fit_on_texts(texts)
        return self
    
    def transform(self, texts, y=None):
        return np.array(self.texts_to_sequences(texts))
        
sequencer = TextsToSequences(num_words=vocab_size)


class Padder(BaseEstimator, TransformerMixin):
    """ Pad and crop uneven lists to the same length. 
    Only the end of lists longernthan the maxlen attribute are
    kept, and lists shorter than maxlen are left-padded with zeros
    
    Attributes
    ----------
    maxlen: int
        sizes of sequences after padding
    max_index: int
        maximum index known by the Padder, if a higher index is met during 
        transform it is transformed to a 0
    """
    def __init__(self, maxlen=500):
        self.maxlen = maxlen
        self.max_index = None
        
    def fit(self, X, y=None):
        self.max_index = pad_sequences(X, maxlen=self.maxlen).max()
        return self
    
    def transform(self, X, y=None):
        X = pad_sequences(X, maxlen=self.maxlen)
        X[X > self.max_index] = 0
        return X

padder = Padder(maxlen)


batch_size = 128
max_features = vocab_size + 1

import tensorflow as tf 
tf.random.set_seed(1221)

def create_model(max_features):
    """ Model creation function: returns a compiled LSTM"""
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Use Keras Scikit-learn wrapper to instantiate a LSTM with all methods
# required by Scikit-learn for the last step of a Pipeline
sklearn_lstm = KerasClassifier(build_fn=create_model, epochs=5, batch_size=batch_size, 
                               max_features=max_features, verbose=1)

# Build the Scikit-learn pipeline
pipeline = make_pipeline(sequencer, padder, sklearn_lstm)

pipeline.fit(X_train, y_train);

print('Computing predictions on test set...')

y_preds = pipeline.predict(X_test)


def model_evaluate(): 
    
    print('Test Accuracy:\t{:0.1f}%'.format(accuracy_score(y_test,y_preds)*100))
    
    #classification report
    print('\n')
    print(classification_report(y_test, y_preds))

    #confusion matrix
    confmat = confusion_matrix(y_test, y_preds)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    
model_evaluate()


# Save the pipeline
joblib.dump(pipeline, 'pipeline_model.joblib')


# We choose a sample from test set
idx = 12
test_text = np.array(X_test)
test_class = np.array(y_test)
text_sample = test_text[idx]
class_names = ['neutral', 'positive', 'negative']
print(text_sample)
print('Probability =', pipeline.predict_proba([text_sample]).round(3))
print('True class: %s' % class_names[test_class[idx]])