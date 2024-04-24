import pandas as pd
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

stop_words = []

train_df = pd.read_csv("EmoSPeech_phase_2_train_public.csv")
test_df = pd.read_csv("EmoSPeech_phase_2_test_public.csv")

# Create a TFIDF Vectorizer using sci-kit. With this, we are going to represent all texts
# as counts of the vocabulary.
vectorizer = TfidfVectorizer (
  analyzer = 'word',
  max_features = 50_000,
  lowercase = False,
  stop_words = stop_words
)


# Get the TF-IDF values from the training set
text_x_train = vectorizer.fit_transform (train_df['transcription'])

# Get the TF-IDF values from the test set
# Note that we apply the TF-IDF learned from the training split
text_x_test = vectorizer.transform (test_df['transcription'])


# We are going to store a baseline per dimension
baselines = {}


# Get a baseline classifier
baselines["label"] = LinearSVC (dual = 'auto')

scaler = MinMaxScaler ()
text_x_train = scaler.fit_transform (text_x_train.A)
text_x_test = scaler.fit_transform (text_x_test.A)

# Train the baseline for this label
baselines["label"].fit (text_x_train, train_df["label"])