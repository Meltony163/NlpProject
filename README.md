# RAG-Based Mental Health Support Chatbot

A local Flask app that integrates multiple NLP modules to provide a grounded, empathetic mental health support chatbot.

## Project Structure

- `app.py` — unified Flask deployment script and request router
- `LanguageIdentification/LImodel.py` — language detection using a saved sklearn pipeline
- `emotion_Classifier/classefier.py` — emotion classification using a transformer model
- `intent_Classifier/IntentClassifier.py` — intent detection using Groq LLM prompting
- `Retreival/Retreival.py` — retrieval module using Qdrant, sentence embeddings, and sparse retrieval
- `.env` — environment configuration for API keys and Qdrant settings
- `requirements.txt` — Python package dependencies

## What it does

The system processes each user message with these steps:

1. Detect user language
2. Predict emotion
3. Classify intent
4. If the intent is `asking_mental_health_question`, run RAG retrieval and generate an answer
5. Otherwise, return a direct canned empathetic response

## API Endpoints

### `GET /`

Returns a simple info page for the running service.

### `POST /api/chat`

Main chatbot endpoint.

Request body (JSON):

```json
{ "text": "I feel anxious and need help." }
```

Response body (JSON):

- `language` — detected language label
- `emotion` — predicted emotion class
- `intent` — detected intent label
- `response` — chatbot response text
- `sources` — retrieved source contexts used for the answer

## Setup

1. Create a Python environment and activate it.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your `.env` file with:

```text
API_KEY="<your Groq API key>"
qdrant_url="<your qdrant endpoint>"
qdrant_api_key="<your qdrant api key>"
```

4. Start the app:

```bash
python app.py
```

The server listens on `http://127.0.0.1:7860`.

## Example usage

### Curl example

```bash
curl -X POST http://127.0.0.1:7860/api/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel stressed and need support."}'
```

### Python example

```python
import requests

response = requests.post(
    'http://127.0.0.1:7860/api/chat',
    json={"text": "I feel stressed and need support."}
)
print(response.json())
```

## Postman / REST client

Use `POST http://127.0.0.1:7860/api/chat` with JSON body:

```json
{
  "text": "أنا تعبان أوي ومش عارف أعمل إيه ممكن تساعدني؟"
}
```

The response should include the detected `language`, predicted `emotion`, predicted `intent`, and a chatbot `response`.

## Troubleshooting

- If the app fails to start, make sure dependencies are installed and `.env` values are correct.
- If Groq or Qdrant are unavailable, the app can still start, but RAG response generation may fall back to a safe message.
- If the language model load fails, verify that `LanguageIdentification/full_pipeline.pkl` and `LabelEncoder.pkl` exist in the `LanguageIdentification` folder.

## Screenshot example

If you want to include a visual example in the repository, save your screenshot as `docs/api-example.png` and add it to the repo. Then use Markdown like:

```markdown
![API request example](docs/api-example.png)
```

This README is intentionally simple so it is easy to present during assessment and debugging.
