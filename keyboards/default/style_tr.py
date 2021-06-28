from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

style_tr = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text="У меня есть картинки")
        ],
        [
            KeyboardButton(text="Сгенерировать картинки")
        ],
        [
            KeyboardButton(text="Сгенрировать 5 картинок для стиля")
        ],
        [
            KeyboardButton(text="Закончить")
        ],
    ],
    resize_keyboard=True
)

style_tr_end = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text="Закончить")
        ],
        ],resize_keyboard=True
)
