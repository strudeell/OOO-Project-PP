import asyncio
import time

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message

from config import token

bot = Bot(token=token)
dp = Dispatcher()

@dp.message(CommandStart())
async def start_message(message: Message):
    await message.answer('Привет, добро пожаловать в чат-бот по распознаванию текста\n'
                         '\n'
                         'Отправьте нам фото с рукописным текстом')

@dp.message(F.photo)
async def photo_handler(message: Message) -> None:
    file_name = f"photos/{message.photo[-1].file_id}.jpg"
    await message.bot.download(file=message.photo[-1].file_id, destination=file_name)
    await message.answer('Спасибо за фото, началась обработка информации')
    time.sleep(7)
    await message.answer('Ваш текст с картинки выглядит так:\n'
                         '\n'
                         'АЛФАВИТ')

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Выход из программы')