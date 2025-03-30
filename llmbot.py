import os
import logging
import time
import telebot
from telebot.types import Message
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
import warnings
from dotenv import load_dotenv
import signal

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
TELEGRAM_API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_TOKEN

if not TELEGRAM_API_TOKEN:
    raise ValueError("TELEGRAM_API_TOKEN is not set in environment variables.")

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
        self.setup_llm()
        self.setup_handlers()
    
    def setup_llm(self):
        """Initialize the LLM and LLMChain"""
        try:
            self.llm = HuggingFaceHub(
                repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                model_kwargs={"temperature": 0.55, "max_length": 512}
            )
            self.prompt = PromptTemplate(
                input_variables=["chat_history", "question"],
                template="""
                Previous conversation:
                {chat_history}
                Human: {question}
                AI Assistant:"""
            )
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
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

    def get_llm_response(self, chat_history_str: str, question: str) -> str:
        try:
            return self.llm_chain.run(chat_history=chat_history_str, question=question)
        except Exception as e:
            logger.error(f"Error during LLM call: {str(e)}")
            return "I'm experiencing technical issues. Please try again later."

    def handle_message(self, message: Message):
        user_id = message.from_user.id
        user_question = message.text
        if user_id not in self.user_chat_histories:
            self.user_chat_histories[user_id] = []

        # chat_history_str = "\n".join([f"Human: {q}\nAI: {a}" for q, a in self.user_chat_histories[user_id]])
        # response = self.get_llm_response(chat_history_str, user_question)
        
        # Format chat history for internal use only (no display to the user)
        chat_history_str = "\n".join([f"{q}\n{a}" for q, a in self.user_chat_histories[user_id]])

        # Get response from LLM
        response = self.get_llm_response(chat_history_str, user_question)

        # Send only the latest response to the user
        self.bot.send_message(user_id, response)

        # Update chat history
        self.user_chat_histories[user_id].append((user_question, response))

        # Maintain max history length
        if len(self.user_chat_histories[user_id]) > MAX_HISTORY_LENGTH:
            self.user_chat_histories[user_id] = self.user_chat_histories[user_id][-MAX_HISTORY_LENGTH:]
        
        self.bot.reply_to(message, response)
        logger.info(f"Responded to user {user_id}")

    def run(self):
        logger.info("Bot is starting...")
        try:
            self.bot.polling(none_stop=True, timeout=60)
        except Exception as e:
            logger.error(f"Error during polling: {str(e)}")
            time.sleep(10)

if __name__ == "__main__":
    bot = AITelegramBot()
    bot.run()
