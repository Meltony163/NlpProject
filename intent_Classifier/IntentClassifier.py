import re
from groq import Groq

from dotenv import load_dotenv
from pathlib import Path
import os

dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

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

  intents = ["greeting","goodbye","gratitude","asking_mental_health_question","out_of_scope"]
  if res not in intents:
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

    if res not in intents:
      return -1
    else:
      return res
  else:
    return res


if __name__ == "__main__":
  user_message = input("User: ")
  print(IntentClassifier(user_message))