from aiogram import types
from aiogram.dispatcher.filters.builtin import CommandHelp

from loader import dp


@dp.message_handler(CommandHelp())
async def bot_help(message: types.Message):
    text = ("Список команд: ",
            "/start - Начать диалог",
            "/style_transfer - Style transfer",
            "/pict - Выбор картинок",
            "/help - Получить справку"
            )
    
    await message.answer("\n".join(text))
