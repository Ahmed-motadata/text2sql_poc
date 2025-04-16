import os
import sys
from typing import Dict, Any
from pydantic import BaseModel, Field
import json
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings.settings import DEFAULT_LLM

from scripts.build_context import build_llm_context


class LLMOutput(BaseModel):
    user_query: str = Field(...)
    llm_response: str = Field(...)
    llm_input_tokens: int = Field(0)
    llm_output_tokens: int = Field(0)
    llm_search_tokens: int = Field(0)
    vector_tokens: int = Field(0)
    
    llm_input_cost: float = Field(0.0)
    llm_output_cost: float = Field(0.0)
    vector_cost: float = Field(0.0)
    

def call_llm(context: str) -> tuple[str, int, int]:
    """
    Calls the LLM using LangChain's ChatOpenAI client and tracks token usage.

    Args:
        context (str): The LLM context to process.

    Returns:
        tuple: (response, input_tokens, output_tokens)
    """
    llm =DEFAULT_LLM
    #ChatOpenAI(model_name="gpt-4o-mini")
    messages = [{"role": "user", "content": context}]
    with get_openai_callback() as cb:
        response = llm.invoke(messages)
        # Extract the response content and ensure it matches the required format
        response_text = response.content.strip()
        input_tokens = cb.prompt_tokens
        output_tokens = cb.completion_tokens
    return response_text, input_tokens, output_tokens

def generate_structured_output(
    user_query: str,
    column_search_method: str = "llm",
    table_search_method: str = "llm"
) -> Dict[str, Any]:
    """
    Generates structured output by invoking build_llm_context and processing LLM response.

    Args:
        user_query (str): The user's query.
        column_search_method (str): Method for column search ("vector" or "llm").
        table_search_method (str): Method for table search ("vector" or "llm").

    Returns:
        Dict containing structured output, token counts, and costs.
    """
    # Step 1: Build LLM context
    context_result = build_llm_context(
        user_query=user_query,
        column_search_method=column_search_method,
        table_search_method=table_search_method
    )

    llm_context = context_result["llm_context"]
    llm_search_tokens = context_result["llm_token_usage"]
    vector_tokens = context_result["vector_token_usage"]

    # Step 2: Call LLM to get response and token counts
    llm_response, llm_input_tokens, llm_output_tokens = call_llm(llm_context)

    # Step 3: Calculate total tokens
    total_llm_input_tokens = llm_search_tokens + llm_input_tokens
    total_llm_tokens = total_llm_input_tokens + llm_output_tokens
    

   # Step 4: Calculate costs
    llm_input_cost_per_million = 0.15  # $0.15 per 1M tokens
    llm_output_cost_per_million = 0.60  # $0.60 per 1M tokens
    vector_cost_per_million = 0.02  # $0.02 per 1M tokens

    llm_input_cost = (total_llm_input_tokens / 1_000_000) * llm_input_cost_per_million
    llm_output_cost = (llm_output_tokens / 1_000_000) * llm_output_cost_per_million
    vector_cost = (vector_tokens / 1_000_000) * vector_cost_per_million
    total_cost = llm_input_cost + llm_output_cost + vector_cost  # Calculate total cost

    # Step 5: Create structured output
    output = LLMOutput(
        user_query=user_query,
        llm_response=llm_response,
        llm_input_tokens=llm_input_tokens,
        llm_output_tokens=llm_output_tokens,
        llm_search_tokens=llm_search_tokens,
        vector_tokens=vector_tokens,
        llm_input_cost=round(llm_input_cost, 8),
        llm_output_cost=round(llm_output_cost, 8),
        vector_cost=round(vector_cost, 8),
        total_cost=total_cost  # Add total cost here
)

    return output.model_dump()

def main():
    user_query = "Show requests with high priority"
    result = generate_structured_output(
        user_query=user_query,
        column_search_method="llm",
        table_search_method="llm"
    )

    # Print structured output
    print("Structured Output:")
    print(json.dumps(result, indent=2))

    # Print token counts and cost breakdown
    print("\nToken Counts and Cost Breakdown:")
    print(f"LLM Search Tokens: {result['llm_search_tokens']}")
    print(f"LLM Input Tokens (Prompt): {result['llm_input_tokens']}")
    print(f"Total LLM Input Tokens (Search + Prompt): {result['llm_search_tokens'] + result['llm_input_tokens']}")
    print(f"LLM Output Tokens: {result['llm_output_tokens']}")
    print(f"Total LLM Tokens: {result['llm_search_tokens'] + result['llm_input_tokens'] + result['llm_output_tokens']}")
    print(f"Vector Tokens: {result['vector_tokens']}")
    
    print("\nCosts:")
    print(f"LLM Input Cost: ${result['llm_input_cost']:.8f}")
    print(f"LLM Output Cost: ${result['llm_output_cost']:.8f}")
    print(f"Vector Cost: ${result['vector_cost']:.8f}")
    print(f"Total Cost: ${result['total_cost']:.8f}")

if __name__ == "__main__":
    main()