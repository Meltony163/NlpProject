import sys
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
import string
import unicodedata
import regex
import joblib
from sklearn.preprocessing import FunctionTransformer
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
sys.modules['__main__'] = sys.modules[__name__]

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


full_pipeline = joblib.load(MODEL_DIR / "full_pipeline.pkl")
LE = joblib.load(MODEL_DIR / "LabelEncoder.pkl")

def LanguageIdentifier(text):
  result = LE.inverse_transform(full_pipeline.predict(np.array([text])))
  return result[0] if len(result) else None


if __name__ == "__main__":
  text = input("Enter your text: ")
  print(LanguageIdentifier(text))