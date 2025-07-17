import os
from litellm import completion, batch_completion

from httpx import HTTPStatusError
from litellm.exceptions import NotFoundError
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List, Tuple

load_dotenv()

temp = 0.01



class DifferenceInspector:
    def __init__(self, judger_model: str, comparer_model: str, temp: float):
        self.judger_model = judger_model
        self.comparer_model = comparer_model
        self.temp = temp

    def run(self, responses: List[str]) -> Tuple[str, str]:
        """
        This should run the comparer on all the responses and generate the pros and cons for each response.
        Then, it should run the judger on the pros and cons and generate the best response.

        It should return the best response.

        Input: List[str]
        Output: str
        """

        comparer_system, comparer_user = comparer_prompt(responses)
        comparer_response = completion(
            model=self.comparer_model,
            messages=[
                {"role": "system", "content": comparer_system},
                {"role": "user", "content": comparer_user},
            ],
            temperature=temp,
        )

        print(f"comparer_response: {comparer_response.choices[0].message.content}")

        judger_system, judger_user = judger_prompt(responses, str(comparer_response.choices[0].message.content))
        judger_response = completion(
            model=self.judger_model,
            messages=[
                {"role": "system", "content": judger_system},
                {"role": "user", "content": judger_user},
            ],
            temperature=temp,
        )
        print(f"comparer_response: {comparer_response.choices[0].message.content}")
        return str(judger_response.choices[0].message.content)

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

list_of_responses = []

comparer_system, comparer_user = comparer_prompt(list_of_responses)
print(f"comparer_system: {comparer_system}")
print(f"comparer_user: {comparer_user}")

comparer_response = completion(
          model="gpt-4.1-2025-04-14",
          messages=[
            {
                "role": "system",
                "content": comparer_system,
            },
            {
                "role": "user",
                "content": comparer_user,
            },
          ],
          temperature=temp,
    )

judger_system, judger_user = judger_prompt(list_of_responses, str(comparer_response.choices[0].message.content))
print(f"judger_system: {judger_system}")
print(f"judger_user: {judger_user}")

judger_response = completion(
          model="gpt-4.1-2025-04-14",
          messages=[
            {
                "role": "system",
                "content": judger_system,
            },
            {
                "role": "user",
                "content": judger_user,
            },
          ],
          temperature=temp,
    )