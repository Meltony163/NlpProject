import re
from groq import Groq

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("API_KEY")

def IntentClassifier(user_message):
  client = Groq(api_key=api_key)

  system_message = {
              "role": "system",
              "content": '''
                User: What are some coping strategies for stress?
                Intent: asking_mental_health_question

                User: Thank you very much
                Intent: gratitude

                User: Who won the World Cup?
                Intent: out_of_scope

                User: Good morning
                Intent: greeting

                User: I have been feeling depressed recently
                Intent: asking_mental_health_question

                User: See you later
                Intent: goodbye

                User: Write a Python program to sort a list
                Intent: out_of_scope

                User: Hello
                Intent: greeting

                User: I appreciate your support
                Intent: gratitude

                User: What's the capital of France?
                Intent: out_of_scope

                User: Bye
                Intent: goodbye

                User: How can I manage anxiety?
                Intent: asking_mental_health_question
                '''}
  user_message = {
              "role": "user",
              "content": f'''
              Classify the intent of the following message.
              {user_message}
              Intent:
              '''}

  chat_completion = client.chat.completions.create( messages=[system_message, user_message], model = "openai/gpt-oss-20b")
  match = re.search(
    r"(greeting|goodbye|gratitude|asking_mental_health_question|out_of_scope)", chat_completion.choices[0].message.content)

  if match:
    res = match.group(1)
  else:
    res = "None"

  intent_to_id = {
    "greeting": 0,
    "goodbye": 1,
    "gratitude": 2,
    "asking_mental_health_question": 3,
    "out_of_scope": 4}

  if res not in intent_to_id.keys():
    response = {}
    response["role"] = chat_completion.choices[0].message.role
    response["content"] = chat_completion.choices[0].message.content

    second_user_message = {"role":"user",
                           "content": "you have to output just the class belongs into greeting, goodbye, gratitude,asking_mental_health_question, out_of_scope. and the output should be  intent: class"}

    chat_completion = client.chat.completions.create( messages=[system_message, user_message, response, second_user_message], model = "openai/gpt-oss-20b")
    match = re.search(r"(greeting|goodbye|gratitude|asking_mental_health_question|out_of_scope)", chat_completion.choices[0].message.content)

    if match:
      res = match.group(1)
    else:
      res = "None"

    if res not in intent_to_id.keys():
      return -1
    else:
      return intent_to_id[res]
  else:
    return intent_to_id[res]


if __name__ == "__main__":
  user_message = input("User: ")
  print(IntentClassifier(user_message))