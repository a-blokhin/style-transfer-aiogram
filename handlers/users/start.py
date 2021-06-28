from aiogram import types
from aiogram.dispatcher.filters.builtin import CommandStart

from loader import dp


@dp.message_handler(CommandStart())
async def bot_start(message: types.Message):
    await message.answer(f"Привет, {message.from_user.full_name}!")
    await message.answer("Бот преднзначен для переноса стилей между картинками и/или стикерами, эта функция включается командой /style_transfer . Если вы хотите найти подходящие для этой задачи картинки, то подбор картинок реализован командой /pict . Картинки в style-transfer отправляйте пожалуйста по одной, если отправляете стикеры, то без прозрачного фона, то есть прямоугольные.")
