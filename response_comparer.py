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
from response_comparer_utils import comparer_prompt, judger_prompt

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
            temperature=self.temp,
        )

        print(f"comparer_response: {comparer_response.choices[0].message.content}")

        judger_system, judger_user = judger_prompt(responses, str(comparer_response.choices[0].message.content))
        judger_response = completion(
            model=self.judger_model,
            messages=[
                {"role": "system", "content": judger_system},
                {"role": "user", "content": judger_user},
            ],
            temperature=self.temp,
        )
        print(f"comparer_response: {comparer_response.choices[0].message.content}")
        return str(judger_response.choices[0].message.content)
    
if __name__ == "__main__":
    