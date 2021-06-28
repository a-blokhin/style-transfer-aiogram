from aiogram.dispatcher.filters import Command, Text
from aiogram.types import Message, ReplyKeyboardRemove
from keyboards.default import pict
from loader import dp
import time
from data.five_im import five_im
from data.picasso import picasso
import random

@dp.message_handler(Command("pict"))
async def show_pict(message: Message):
    await message.answer("Выберите картинку", reply_markup=pict)


@dp.message_handler(Text(equals=["Человек", "Кисек", "Лошадка", "Пикассо", "5 картинок для стиля", "Сгенрировать 5 картинок для стиля", "Закончить"]))
async def get_food(message: Message):
    if message.text == "Человек":
        await message.answer_photo(photo='https://thispersondoesnotexist.com/image?'+str(time.time()))
    elif message.text == "Кисек":
        await message.answer_photo(photo='https://thiscatdoesnotexist.com/?'+str(time.time()))
    elif message.text == "Лошадка":
        await message.answer_photo(photo='https://thishorsedoesnotexist.com/?'+str(time.time()))
    elif message.text == "Пикассо":
        n = random.randint(0,5)
        y=''
        for i, x in enumerate(picasso):
            if i == n:
                y = x
        z = picasso[y][random.randint(0,len(picasso)-1)]
        await message.answer_photo(photo='https://gallerix.ru/pic/_EX/'+y+'/'+z+'.jpeg')
    elif message.text == "5 картинок для стиля" or message.text == "Сгенрировать 5 картинок для стиля":
        for im in five_im:
            await message.answer_photo(photo=im)
    elif message.text == "Закончить":
        await message.answer(text = "На этом закончили", reply_markup=ReplyKeyboardRemove())




