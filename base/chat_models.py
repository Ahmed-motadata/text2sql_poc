import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
 
load_dotenv()
 
 
chat_gpt4o = ChatOpenAI(
    model="gpt-4o",  
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
 
chat_gpt4 = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
 
chat_gpt_o3mini = ChatOpenAI(
    model="gpt-o3-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
 
chat_groq_llama = ChatGroq(
    model="llama-3.3-70b-versatile",  
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)
 
chat_groq_llama_70b = ChatGroq(
    model="llama3-70b-8192",  
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"))
 
chat_groq_deepseek = ChatGroq(
    model="deepseek-r1-distill-llama-70b",  
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"))
 
 
 
chat_gemini_flash = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",  
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

chat_gemini_flash_lite = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite-001",  
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
 