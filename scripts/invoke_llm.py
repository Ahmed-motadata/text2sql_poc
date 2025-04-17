import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
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
    # raw_response = llm.invoke(llm_context)
    # print(raw_response)
    raw_output = llm.invoke(llm_context)
    token_usage = raw_output.response_metadata['token_usage']
    return {
        'input_tokens': token_usage.get('prompt_tokens', 0),
        'output_tokens': token_usage.get('completion_tokens', 0),
        'total_tokens': token_usage.get('total_tokens', 0)
    }
 
def calculate_price(token_counts):
    """
    Calculate the price based on token usage.
    
    Args:
        token_counts: Dictionary containing input, output, and vector token counts
        
    Returns:
        Dictionary with prices for input, output, vector, and total
    """
    # Pricing per million tokens
    INPUT_PRICE_PER_MILLION = 0.15  # $0.15 per million input tokens
    OUTPUT_PRICE_PER_MILLION = 0.60  # $0.60 per million output tokens
    VECTOR_PRICE_PER_MILLION = 0.020  # $0.020 per million vector tokens
    
    # Calculate prices
    input_price = (token_counts['input_tokens'] / 1_000_000) * INPUT_PRICE_PER_MILLION
    output_price = (token_counts['output_tokens'] / 1_000_000) * OUTPUT_PRICE_PER_MILLION
    vector_price = (token_counts['vector_tokens'] / 1_000_000) * VECTOR_PRICE_PER_MILLION
    total_price = input_price + output_price + vector_price
    
    return {
        'input_price': input_price,
        'output_price': output_price,
        'vector_price': vector_price,
        'total_price': total_price
    }
 
if __name__ == "__main__":
    user_query = input("Please enter your query: ")
    result = build_llm_context(user_query)
    llm_context = result["llm_context"]
    vector_token_count = result["vector_token_count"]
    llm_token_usage = result["llm_token_usage"]
    token_usage = result.get("token_usage", {})
    
    # Call LLM to generate SQL
    sql_gen_tokens = call_llm(llm_context)
    
    # Print detailed token usage information
    print(f"\n\n=== Token Usage Breakdown ===")
    
    # Table retrieval token usage
    table_usage = token_usage.get("table", {})
    print(f"Table Retrieval:")
    print(f"  - Input tokens:  {table_usage.get('input_tokens', 0)}")
    print(f"  - Output tokens: {table_usage.get('output_tokens', 0)}")
    print(f"  - Total:         {table_usage.get('total_tokens', 0)}")
    
    # Column retrieval token usage
    column_usage = token_usage.get("column", {})
    print(f"Column Retrieval:")
    print(f"  - Input tokens:  {column_usage.get('input_tokens', 0)}")
    print(f"  - Output tokens: {column_usage.get('output_tokens', 0)}")
    print(f"  - Total:         {column_usage.get('total_tokens', 0)}")
    
    # SQL generation token usage
    print(f"SQL Generation:")
    print(f"  - Input tokens:  {sql_gen_tokens.get('input_tokens', 0)}")
    print(f"  - Output tokens: {sql_gen_tokens.get('output_tokens', 0)}")
    print(f"  - Total:         {sql_gen_tokens.get('total_tokens', 0)}")
    
    # Calculate total tokens by category
    total_input_tokens = (
        table_usage.get('input_tokens', 0) + 
        column_usage.get('input_tokens', 0) + 
        sql_gen_tokens.get('input_tokens', 0)
    )
    
    total_output_tokens = (
        table_usage.get('output_tokens', 0) + 
        column_usage.get('output_tokens', 0) + 
        sql_gen_tokens.get('output_tokens', 0)
    )
    
    # Summary
    total_tokens = vector_token_count + llm_token_usage + sql_gen_tokens.get('total_tokens', 0)
    print(f"\n=== Summary ===")
    print(f"Vector Search Token Count: {vector_token_count}")
    print(f"Input Tokens:              {total_input_tokens}")
    print(f"Output Tokens:             {total_output_tokens}")
    print(f"Total Tokens:              {total_tokens}")
    
    # Price calculation
    token_counts = {
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'vector_tokens': vector_token_count
    }
    
    prices = calculate_price(token_counts)
    
    print(f"\n=== Cost Calculation ===")
    print(f"Input tokens:  {total_input_tokens:,} tokens = ${prices['input_price']:.6f}")
    print(f"Output tokens: {total_output_tokens:,} tokens = ${prices['output_price']:.6f}")
    print(f"Vector tokens: {vector_token_count:,} tokens = ${prices['vector_price']:.6f}")
    print(f"Total cost:    ${prices['total_price']:.6f}")