import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
import string
import unicodedata
import regex
import joblib
from sklearn.preprocessing import FunctionTransformer

class RemovingPunctation(BaseEstimator, TransformerMixin):
    def __init__(self):
      pass

    def fit(self, X, y=None):
      return self

    def transform(self, X):
      res = X.copy()
      if not isinstance(res,np.ndarray):
        res = res.to_numpy()

      res = np.array([
          ''.join(
              ' ' if unicodedata.category(ch).startswith('P') else ch
              for ch in str(text)
          )
          for text in res
      ])
      return res

class RemovingEmojis(BaseEstimator, TransformerMixin):
    def __init__(self):
      pass

    def fit(self, X, y=None):
      return self

    def transform(self, X):
      res = X.copy()
      if not isinstance(res,np.ndarray):
        res = res.to_numpy()

      res = np.array([regex.sub(r'\p{Emoji}', '', str(x)) for x in res])
      return res

class RemovingNumbers(BaseEstimator, TransformerMixin):
    def __init__(self):
      pass

    def fit(self, X, y=None):
      return self

    def transform(self, X):
      res = X.copy()
      if not isinstance(res,np.ndarray):
        res = res.to_numpy()

      res = np.array([re.sub(r'\d', ' ', str(x)) for x in res])
      return res

class ShrinkSpaces(BaseEstimator, TransformerMixin):
    def __init__(self):
      pass

    def fit(self, X, y=None):
      return self

    def transform(self, X):
      res = X.copy()
      if not isinstance(res,np.ndarray):
        res = res.to_numpy()

      res = np.array([re.sub(r'\s+', ' ', str(x)) for x in res])
      return res

def to_lower(X):
    return np.array([str(x).lower() for x in X], dtype=object)

LT = FunctionTransformer(to_lower)


full_pipeline = joblib.load("full_pipeline.pkl")
LE = joblib.load("LabelEncoder.pkl")

def LanguageIdentifier(text):
  return LE.inverse_transform(full_pipeline.predict(np.array([text])))


if __name__ == "__main__":
  text = input("Enter your text: ")
  print(LanguageIdentifier(text))