# Emotion Detection Transformer

This repository contains a PyTorch-based emotion classification model built with a small transformer encoder. It loads a pretrained model checkpoint and vocabulary, then predicts the emotion expressed in a text input.

## Files

- `main.py` - example entry point that imports `prediction` from `classefier.py` and prints a sample prediction.
- `classefier.py` - model definition, tokenizer, vocabulary loading, checkpoint loading, and `prediction(text)` function.
- `best_scratch_transformer.pt` - pretrained model weights.
- `vocab.dic` - saved vocabulary mapping token strings to integer IDs.
- `__init__.py` - empty package initializer.

## Requirements

- Python 3.8+
- PyTorch

Install PyTorch using the official instructions for your system. For CPU-only use:

```bash
pip install torch torchvision torchaudio
```

## Usage

From the repository root, run:

```bash
python main.py
```

This will execute `main.py`, which calls `prediction("moamen is terrified")` and prints the predicted emotion.

## Prediction API

Use the `prediction(text)` function from `classefier.py` to classify text.

Example:

```python
from classefier import prediction

result = prediction("I am very happy")
print(result)
```

## Notes

- The model expects input text tokenized with whitespace splitting.
- The tokenizer adds `<CLS>` and `<SEP>` tokens, pads inputs to length 64, and uses a saved vocabulary.
- The predicted classes are:
  - `sadness`
  - `joy`
  - `love`
  - `anger`
  - `fear`
  - `surprise`

If you want to use the model in a different script, import `prediction` from `classefier` and ensure `best_scratch_transformer.pt` and `vocab.dic` are present in the same directory.