from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

from data.config import URL_APPLES, URL_PEAR
from keyboards.inline.callback_datas import buy_callback

# choice = InlineKeyboardMarkup(
#     inline_keyboard=[
#         [
#             InlineKeyboardButton(text="Купить грушу", callback_data=buy_callback.new(
#                 item_name="pear", quantity=1
#             )),
#             InlineKeyboardButton(text="Купить яблоко", callback_data="buy:apple:5")
#         ],
#         [
#             InlineKeyboardButton(text="Отмена", callback_data="cancel")
#         ]
#     ]
# )

choice = InlineKeyboardMarkup(row_width=2)

buy_pear = InlineKeyboardButton(text="Купить грушу", callback_data=buy_callback.new(
                 item_name="pear", quantity=1
             ))

choice.insert(buy_pear)
buy_apples = InlineKeyboardButton(text="Купить яблоко", callback_data="buy:apple:5")
choice.insert(buy_apples)

cancel_button = InlineKeyboardButton(text="Отмена", callback_data="cancel")
choice.insert(cancel_button)

pear_keyboard = InlineKeyboardMarkup(inline_keyboard=[
    [
        InlineKeyboardButton(text="Купи тут", url=URL_APPLES)
    ]
])
apples_keyboard = InlineKeyboardMarkup(inline_keyboard=[
    [
        InlineKeyboardButton(text="Купи тут", url=URL_PEAR)
    ]
])
