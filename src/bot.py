import joblib
from loader import NeighbourCample
from telebot import types
from telebot.types import Message
import telebot

TG_BOT_TOKEN = ("token")
bot = telebot.TeleBot(TG_BOT_TOKEN)
pipe = joblib.load('pipeline.pkl')


@bot.message_handler(commands=['start'])
def start(m):
    bot.send_message(m.chat.id, 'Talk to me, Baby')


@bot.message_handler(func=lambda message: True)
def magic_kick(message):
    bot.send_message(message.chat.id, pipe.predict([message.text.lower()])[0])


bot.polling()
