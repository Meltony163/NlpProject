"""
Unified RAG-based mental health support chatbot.
Integrates:
- Language detection
- Emotion classification
- Intent routing
- Retrieval-augmented generation using Qdrant and Groq
"""

import os
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from LanguageIdentification.LImodel import LanguageIdentifier
from emotion_Classifier.classefier import prediction as EmotionClassifier
from intent_Classifier.IntentClassifier import IntentClassifier
from Retreival.Retreival import retreive

try:
    from groq import Groq
except ImportError:
    Groq = None

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("API_KEY")
client = None
if Groq is not None and API_KEY:
    try:
        client = Groq(api_key=API_KEY)
    except Exception as exc:
        print("Warning: failed to initialize Groq client:", exc)

INTENT_LABELS = {
    0: "greeting",
    1: "goodbye",
    2: "gratitude",
    3: "asking_mental_health_question",
    4: "out_of_scope",
}

RULE_RESPONSES = {
    "greeting": "Hello! I am here to support you. How are you feeling today?",
    "goodbye": "Take care. If you need to talk again, I am here for you.",
    "gratitude": "You are welcome. I am glad I can support you.",
    "out_of_scope": "I am designed to help with mental health support. Can you share more about how you are feeling?",
}

app = Flask(__name__)


def build_rag_prompt(user_text, emotion, language, retrieved):
    sections = []
    for index, (context, response) in enumerate(retrieved[:4], start=1):
        sections.append(f"Knowledge {index}: {context}\nResponse {index}: {response}")

    context_block = "\n\n".join(sections) if sections else "No retrieval contexts were found."
    return (
        "You are a compassionate mental health support assistant. "
        "Use the retrieved counseling knowledge to answer the user in a grounded and empathetic way. "
        "If the user needs anxiety, depression, stress, or crisis support, give kind and safe guidance. "
        "Answer the query in the same language that the user used whenever possible.\n\n"
        f"User language: {language}\n"
        f"User emotion: {emotion}\n"
        f"User question: {user_text}\n\n"
        "Retrieved knowledge:\n"
        f"{context_block}\n\n"
        "Provide a clear and empathetic answer grounded in the knowledge above. "
        "If you cannot answer from the retrieved content, still respond kindly and safely."
    )


def generate_rag_response(user_text, emotion, language, retrieved):
    if not client:
        return (
            "I am unable to access the RAG generation service right now, "
            "but I am still here to listen. Please tell me more about how you are feeling."
        )

    system_message = {
        "role": "system",
        "content": (
            "You are an empathetic mental health chatbot. Answer with compassion, grounding, and respect. "
            "When possible, respond in the user's language."
        ),
    }
    user_message = {
        "role": "user",
        "content": build_rag_prompt(user_text, emotion, language, retrieved),
    }

    try:
        completion = client.chat.completions.create(
            messages=[system_message, user_message],
            model="openai/gpt-oss-20b",
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as exc:
        print("RAG generation error:", exc)
        answer = (
            "I am having trouble generating a response from the RAG system right now. "
            "Please know I still care about what you are experiencing."
        )

    return answer


def build_response(user_text):
    language = LanguageIdentifier(user_text) or "unknown"
    emotion = EmotionClassifier(user_text)
    intent_id = IntentClassifier(user_text)
    intent = INTENT_LABELS.get(intent_id, "out_of_scope")

    if intent != "asking_mental_health_question":
        response = RULE_RESPONSES.get(intent, RULE_RESPONSES["out_of_scope"])
        return {
            "language": language,
            "emotion": emotion,
            "intent": intent,
            "response": response,
            "sources": [],
        }

    retrieved = retreive(user_text)
    response = generate_rag_response(user_text, emotion, language, retrieved)
    sources = [context for context, _ in retrieved]

    return {
        "language": language,
        "emotion": emotion,
        "intent": intent,
        "response": response,
        "sources": sources,
    }


@app.route("/", methods=["GET"])
def home():
    return (
        "<h1>RAG Mental Health Support Chatbot</h1>"
        "<p>Use <code>POST /api/chat</code> with JSON <code>{\"text\": \"your message\"}</code>.</p>"
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_text = (data or {}).get("text", "")
    if not isinstance(user_text, str) or not user_text.strip():
        return jsonify({"error": "Please provide a non-empty 'text' field."}), 400

    try:
        payload = build_response(user_text.strip())
        return jsonify(payload)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)