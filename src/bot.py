import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from loader import prepare_data, NeighbourCample
from telebot import types
from telebot.types import Message
import telebot

TG_BOT_TOKEN = ('1783843952:AAGkKSVzqjo5HlwmDDABI4t3AEcPqw54cRw')
bot = telebot.TeleBot(TG_BOT_TOKEN)

smal_matr, chandler, vector, svd = prepare_data()
ns = NeighbourCample()
ns.fit(smal_matr, chandler.script)
pipe = make_pipeline(vector, svd, ns)

@bot.message_handler(commands=['start'])
def start(m):
    bot.send_message(m.chat.id, 'Talk to me, Baby')


@bot.message_handler(func=lambda message: True)
def magic_kick(message):
    bot.send_message(message.chat.id, pipe.predict([message.text.lower()])[0])


bot.polling()