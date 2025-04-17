import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
 
load_dotenv()
 
# Initialize OpenAI models
OPENAI_4 = ChatOpenAI(model_name="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_4O_MINI = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_03_MINI = ChatOpenAI(model_name="gpt-o3-mini", openai_api_key=os.getenv('OPENAI_API_KEY'))

# # Initialize GROQ models conditionally - handle Pydantic version conflicts
# try:
#     GROQ_LLAMA_3_3_70b = init_chat_model("llama-3.3-70b-versatile", model_provider="groq", api_key=os.getenv("GROQ_API_KEY"))
#     GROQ_LLAMA_3_70b = init_chat_model("llama3-70b-8192", model_provider="groq", api_key=os.getenv('GROQ_API_KEY'))
#     GROQ_DEEPSEEK_R1 = init_chat_model("deepseek-r1-distill-llama-70b", model_provider="groq", api_key=os.getenv('GROQ_API_KEY'))
# except Exception as e:
#     print(f"Warning: GROQ models initialization failed. Using None instead. Error: {e}")
#     # Provide fallback None values to prevent import errors
#     GROQ_LLAMA_3_3_70b = None 
#     GROQ_LLAMA_3_70b = None
#     GROQ_DEEPSEEK_R1 = None

# # Initialize Google models conditionally
# try:
#     GEMINI_FLASH = init_chat_model("gemini-2.0-flash-001",model_provider="google_genai", api_key=os.getenv('GOOGLE_API_KEY'))
#     GEMINI_FLASH_LITE = init_chat_model("gemini-2.0-flash-lite-001",model_provider="google_genai", api_key=os.getenv('GOOGLE_API_KEY'))
# except Exception as e:
#     print(f"Warning: Gemini models initialization failed. Using None instead. Error: {e}")
#     GEMINI_FLASH = None
#     GEMINI_FLASH_LITE = None