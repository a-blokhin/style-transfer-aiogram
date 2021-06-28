from PIL import Image
from aiogram.dispatcher.filters import Command, Text, state
from aiogram.types import Message, ReplyKeyboardRemove
from keyboards.default import style_tr,style_tr_end
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Command
from loader import dp, bot, event_loop
from st_tr_class2 import run_style_transfer
from states.test import Test
from io import BytesIO

@dp.message_handler(Command("style_transfer"))
async def enter_test(message: Message):
    await message.answer("Есть ли у вас фотографии которые вы хотите преобразовать?", reply_markup=style_tr)

@dp.message_handler(Text("У меня есть картинки"))
async def enter_test(message: Message):
    await message.answer("Отправьте картинку стиль которой хотели бы поменять", reply_markup=style_tr_end)
    await Test.Q1.set()

@dp.message_handler(content_types=['photo', 'sticker'], state = Test.Q1)
async def answer_q1(message: Message, state: FSMContext, loop=event_loop):
    global content_im
    if message.sticker == None:
        content_im = await bot.get_file(message.photo[-1].file_id)
    else:
        content_im = await bot.get_file(message.sticker.file_id)

    await message.answer("Отправьте картинку у которой хотели бы взять стиль", reply_markup=style_tr_end)
    await Test.Q2.set()


@dp.message_handler(content_types=['photo', 'sticker'], state=Test.Q2)
async def answer_q2(message: Message, state: FSMContext, loop=event_loop):
    await state.finish()
    await message.answer(text="Картинки могут обрабатываться до 5 минут", reply_markup=ReplyKeyboardRemove())

    if message.sticker == None:
        style = await bot.get_file(message.photo[-1].file_id)
    else:
        style = await bot.get_file(message.sticker.file_id)

    bytes_style = BytesIO()
    bytes_content = BytesIO()
    await content_im.download(bytes_content)
    await style.download(bytes_style)
    bytes_style.seek(0)
    bytes_content.seek(0)
    cont_im = Image.open(bytes_content)
    style = Image.open(bytes_style)

    input =await loop.create_task(run_style_transfer().run_style_transfer(cont_im, style, loop = loop))
    await message.answer_photo(input)


@dp.message_handler( state=Test.Q1)
async def answer_q1(message: Message, state: FSMContext):

    if message.text == "Закончить":
        await message.answer(text = "На этом закончили", reply_markup=ReplyKeyboardRemove())
        await state.finish()
    else:
        await message.answer(text = "Нужно отправить картинку")


@dp.message_handler(state=Test.Q2)
async def answer_q1(message: Message, state: FSMContext):
    if message.text == "Закончить":
        await message.answer(text="На этом закончили", reply_markup=ReplyKeyboardRemove())
        await state.finish()
    else:
        await message.answer(text="Нужно отправить картинку")


