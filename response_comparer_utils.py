from typing import List, Tuple
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

def comparer_prompt(responses: List[str]) -> Tuple[str, str]:
    system_output = """
    You are an assistant that takes in the multiple responses from the user,
    and then writes unique pros and cons for each of the models' responses relative to other responses.

    ---
    
    Here's a basic example below. Please use this output structure:

    User input:
    [response 1]
    'hi'

    [response 2]
    'hello there'

    Your output:
    [response 1]
    Pros:
    1. shorter
    Cons:
    2. cold

    [response 2]
    Pros:
    1. more formal
    Cons:
    2. longer

    ---
    """
    user_output = ""
    for i in range(len(responses)):
        user_output += f"[response {i + 1}]\n\'" + responses[i] + "\'\n\n"

    return system_output, user_output

def judger_prompt(responses: List[str], pros_and_cons: str) -> Tuple[str, str]:
    system_output = """
    You are an assistant that takes in the multiple responses and the differences (pros and cons) from the user.
    Using this, come up with the best response possible combining these responses. 

    ---
    
    Here's a basic example below. Please use this output structure:

    User input:
    [response 1]
    'hi'

    [response 2]
    'hello there'

    [pros and cons]
    [response 1]
    Pros:
    1. shorter
    Cons:
    2. cold

    [response 2]
    Pros:
    1. more formal
    Cons:
    2. longer

    Your output:
    Hi there

    ---
    """
    user_output = ""
    for i in range(len(responses)):
        response = str(responses[i].choices[0].message.content)
        user_output += f"[response {i + 1}]\n\'" + response + "\'\n\n"
    user_output += f"[pros and cons]\n" + pros_and_cons
    return system_output, user_output