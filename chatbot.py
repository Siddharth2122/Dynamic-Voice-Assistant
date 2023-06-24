from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import time
time.clock = time.time

chatbot = ChatBot("bot")
#trainer = ChatterBotCorpusTrainer(chatbot)
#trainer.train("chatterbot.corpus.english")

def main():
   while True:
     query = input(">> ")
     print(chatbot.get_response(query))
     if "exit" in query:
        break