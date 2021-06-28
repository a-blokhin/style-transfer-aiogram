from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

pict = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text="Человек"),
            KeyboardButton(text="Кисек")
        ],
        [
            KeyboardButton(text="Лошадка"),
            KeyboardButton(text="Пикассо")
        ],
        [
            KeyboardButton(text="5 картинок для стиля"),
            KeyboardButton(text="Закончить")
        ],
    ],
    resize_keyboard=True
)
