from ddgs import DDGS
from pydantic import BaseModel
from langchain.tools import tool, ToolRuntime

class SearchResults(BaseModel):
    title: str
    href: str
    body: str

@tool
def search_internet(query: str, max_results: int = 5):
    """
    Searches the internet using DuckDuckGo.
    
    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return. Defaults to 5.
        
    Returns:
        list: A list of dictionaries containing the search results (title, href, body).
    """
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results, backend="google, bing")
    pdResult = [SearchResults(**result) for result in results]
    return pdResult


@tool
def summarize_conversation(runtime: ToolRuntime) -> str:
    """Summarize the conversation so far."""
    messages = runtime.state["messages"]
    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")

    return f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses, and {tool_msgs} tool results"


@tool
def calculate(expression: str) -> str:
    """Safely evaluate mathematical expressions using AST parsing.
    
    Args:
        expression: Mathematical expression string to evaluate
    
    Returns:
        String with calculation result or error message
        
    Supports arithmetic, math functions (sin, cos, sqrt, log), and constants (pi, e)
    """
    try:
        import ast
        import math
        
        # Parse the expression into an AST
        node = ast.parse(expression, mode='eval')
        
        # Define allowed node types for safe evaluation
        allowed_nodes = {
            ast.Expression, ast.Constant, ast.Num, ast.BinOp,
            ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
            ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Name,
            ast.Load, ast.Call, ast.keyword
        }
        
        # Define allowed functions
        allowed_functions = {
            'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum, 'pow': pow,
            'sqrt': lambda x: x ** 0.5,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'log': math.log, 'pi': math.pi, 'e': math.e
        }
        
        # Check if all nodes are allowed
        for node_item in ast.walk(node):
            if type(node_item) not in allowed_nodes:
                return f"Error: Unsupported operation: {type(node_item).__name__}"
        
        # Evaluate safely
        result = eval(compile(node, '<string>', 'eval'), 
                     {"__builtins__": {}}, allowed_functions)
        
        return f"{expression} = {result}"
    
    except SyntaxError:
        return "Error: Invalid mathematical expression"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error: {str(e)}"



