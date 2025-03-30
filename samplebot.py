from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

TOKEN: Final = '8142158412:AAEVhy1H_X2ZxT2wEqun_OxqD7SKWqnFqPo'
BOT_USERNAME: Final = 'wanderwbot'

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello! I am WanderWBot, your personal travel assistant. How can I help you today?')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('I am here to help you with your travel needs. Just type /start to begin.')

async def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('This is a custom command.')

# Responses to messages
def handle_response(text: str) -> str:
    processed: str = text.lower()
    
    if 'hello' in processed:
        return 'Hey There!'
    if 'how are you' in processed:
        return 'Great'
    
    if 'karo kuch' in processed:
        return 'hum karte hai kuch'
    
    return 'Clear bol Clear!'

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text
    
    print(f'User: {update.message.chat.id} Message Type: {message_type}')
    
    if message_type == 'PRIVATE':  # Changed from 'private' to 'PRIVATE'
        response: str = handle_response(text)
        await update.message.reply_text(response)
    elif message_type == 'GROUP':  # Changed from 'group' to 'GROUP' and changed if to elif
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME, '')
            response: str = handle_response(new_text)
            await update.message.reply_text(response)  # Added this line
    else: 
        response: str = handle_response(text)
        await update.message.reply_text(response)
    
    print(f'Bot: {response}')
    
async def error_handler(update, context):  # Removed type hints to make it more flexible
    print(f'Update {update} Error: {context.error}')
    
if __name__ == '__main__':
    print("Bot is starting")
    app = Application.builder().token(TOKEN).build()
    
    # commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('custom', custom_command))
    
    # Use TEXT filter instead of Text
    app.add_handler(MessageHandler(filters.TEXT, message_handler))
    
    app.add_error_handler(error_handler)
    print("Bot is running")
    app.run_polling(poll_interval=3)
