import os
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from LanguageIdentification.LImodel import LanguageIdentifier
from emotion_Classifier.classefier import prediction as EmotionClassifier
from intent_Classifier.IntentClassifier import IntentClassifier
from Retreival.Retreival import retreive
from Translator.Tmodel import translate

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

RULE_RESPONSES = {
    "greeting": "Hello! I am here to support you. How are you feeling today?",
    "goodbye": "Take care. If you need to talk again, I am here for you.",
    "gratitude": "You are welcome. I am glad I can support you.",
    "out_of_scope": "I am designed to help with mental health support. Can you share more about how you are feeling?",
}

app = Flask(__name__)


def build_system_prompt(emotion, language, retrieved):
    sections = []
    for index, (context, response) in enumerate(retrieved[:4], start=1):
        sections.append(
            f"Example {index}:\n"
            f"  Patient: {context}\n"
            f"  Counselor: {response}"
        )

    examples_block = "\n\n".join(sections) if sections else "No examples available."

    return (
        "You are a compassionate mental health support assistant.\n"
        "Use the following counseling examples to guide your response in a grounded and empathetic way.\n"
        "If the user needs anxiety, depression, stress, or crisis support, give kind and safe guidance.\n"
        f"Answer the query in the same language that the user used (user language: {language}).\n"
        f"The user's detected emotion is: {emotion}.\n\n"
        "--- Retrieved Counseling Examples ---\n"
        f"{examples_block}\n"
        "--- End of Examples ---\n\n"
        "Now respond to the user's message below with a clear and empathetic answer "
        "grounded in the examples above. "
        "If you cannot answer from the examples, still respond kindly and safely."
    )


def generate_rag_response(user_text, emotion, language, retrieved):
    if not client:
        return (
            "I am unable to access the RAG generation service right now, "
            "but I am still here to listen. Please tell me more about how you are feeling."
        )

    system_message = {
        "role": "system",
        "content": build_system_prompt(emotion, language, retrieved),
    }
    user_message = {
        "role": "user",
        "content": user_text,
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

    if language != "en":
        try:
            translated_text = translate(user_text, language, "en")
        except Exception as exc:
            print(f"Translation failed ({language} -> en):", exc)
            translated_text = user_text
    else:
        translated_text = user_text

    emotion = EmotionClassifier(translated_text)
    intent = IntentClassifier(translated_text)

    if intent != "asking_mental_health_question":
        response = RULE_RESPONSES.get(intent, RULE_RESPONSES["out_of_scope"])
        return {
            "language": language,
            "emotion": emotion,
            "intent": intent,
            "response": response,
            "sources": [],
        }

    retrieved = retreive(translated_text)
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