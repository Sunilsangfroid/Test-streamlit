import os
import logging
import time
import telebot
from telebot.types import Message
from dotenv import load_dotenv
import warnings
import signal
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ignore warnings
warnings.filterwarnings("ignore")

# Constants
MAX_HISTORY_LENGTH = 5
MAX_RESPONSE_LENGTH = 4000
TELEGRAM_API_TOKEN = os.getenv("TELEGRAM_API_TOKEN","8142158412:AAEVhy1H_X2ZxT2wEqun_OxqD7SKWqnFqPo")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY","AIzaSyBFH-nK4_QMO12KQunkwo79RSNLVnBjTQM")

if not TELEGRAM_API_TOKEN or not GEMINI_API_KEY:
    raise ValueError("Missing TELEGRAM_API_TOKEN or GEMINI_API_KEY in environment variables.")

# Graceful shutdown
def shutdown_handler(signum, frame):
    logger.info("Shutting down the bot gracefully...")
    exit(0)
signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

class AITelegramBot:
    def __init__(self):
        self.bot = telebot.TeleBot(TELEGRAM_API_TOKEN)
        self.user_chat_histories = {}
        self.setup_gemini()
        self.setup_handlers()

    def setup_gemini(self):
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel("gemini-1.5-pro")
            logger.info("Gemini model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {e}")
            raise

    def setup_handlers(self):
        @self.bot.message_handler(commands=['start'])
        def send_welcome(message: Message):
            self.handle_start(message)
        
        @self.bot.message_handler(commands=['clear'])
        def clear_history(message: Message):
            self.handle_clear(message)
        
        @self.bot.message_handler(commands=['help'])
        def help_command(message: Message):
            self.handle_help(message)
        
        @self.bot.message_handler(func=lambda message: True)
        def handle_messages(message: Message):
            self.handle_message(message)

    def handle_start(self, message: Message):
        user_id = message.from_user.id
        self.user_chat_histories[user_id] = []
        self.bot.reply_to(message, "Welcome! I'm your AI assistant. You can ask me anything. Use /clear to reset our conversation or /help for more info.")

    def handle_clear(self, message: Message):
        user_id = message.from_user.id
        self.user_chat_histories[user_id] = []
        self.bot.reply_to(message, "Your conversation history has been cleared.")

    def handle_help(self, message: Message):
        self.bot.reply_to(message, "I am an AI assistant. Use /start to begin, /clear to reset our conversation, or just ask me anything.")

    def get_gemini_response(self, chat_history_str: str, question: str) -> str:
        try:
            prompt = f"""
            Don't do any formatting, use plain text with bullet points instead.
            Previous conversation:
            {chat_history_str}
            Human: {question}
            AI Assistant:"""
            response = self.model.generate_content(prompt)
            if not response or not response.text:
                return "Sorry, I couldn't generate a response."
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error during Gemini call: {e}")
            return "I'm experiencing technical issues. Please try again later."

    def send_message_safely(self, user_id, response):
        if len(response) > MAX_RESPONSE_LENGTH:
            for i in range(0, len(response), MAX_RESPONSE_LENGTH):
                self.bot.send_message(user_id, response[i:i+MAX_RESPONSE_LENGTH])
        else:
            self.bot.send_message(user_id, response)

    def handle_message(self, message: Message):
        user_id = message.from_user.id
        user_question = message.text
        if user_id not in self.user_chat_histories:
            self.user_chat_histories[user_id] = []

        chat_history_str = "\n".join([f"Human: {q}\nAI: {a}" for q, a in self.user_chat_histories[user_id]])

        response = self.get_gemini_response(chat_history_str, user_question)

        self.send_message_safely(user_id, response)

        self.user_chat_histories[user_id].append((user_question, response))

        if len(self.user_chat_histories[user_id]) > MAX_HISTORY_LENGTH:
            self.user_chat_histories[user_id] = self.user_chat_histories[user_id][-MAX_HISTORY_LENGTH:]
        
        logger.info(f"Responded to user {user_id}")

    def run(self):
        logger.info("Bot is starting...")
        while True:
            try:
                self.bot.polling(none_stop=True, timeout=60)
            except Exception as e:
                logger.error(f"Error during polling: {e}")
                time.sleep(10)

if __name__ == "__main__":
    bot = AITelegramBot()
    bot.run()

