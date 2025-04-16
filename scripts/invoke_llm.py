import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import json
from pydantic import BaseModel
from settings.settings import DEFAULT_LLM
from scripts.build_context import build_llm_context


class LLMOutput(BaseModel):
    sql_query: str

def call_llm(llm_context: str) -> LLMOutput:
    """
    Call the default LLM using llm_context and return the structured SQL query as LLMOutput.
    Always returns a JSON object with only the sql_query field.
    """
    llm = DEFAULT_LLM
    structured_llm_output = llm.with_structured_output(LLMOutput)
    response = structured_llm_output.invoke(llm_context)
    print(response.sql_query)

if __name__ == "__main__":
    user_query = input("Please enter your query: ")
    result = build_llm_context(user_query)
    llm_context = result["llm_context"]
    call_llm(llm_context)
