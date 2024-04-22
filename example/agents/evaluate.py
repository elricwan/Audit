"""
=======
result evaluator
=======
@date: 2024-4-11
"""
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
print(sys.path)

import numpy as np
from xyz.node.agent import Agent
from xyz.node.basic.llm_agent import LLMAgent
from xyz.utils.llm.openai_client import OpenAIClient
import os
import json
import re
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


class Evalutor(Agent):
    def __init__(self):
        
        super().__init__()  

        self.openai_agent = OpenAIClient(api_key=OPENAI_API_KEY, model='gpt-4-0125-preview', temperature=0.7, top_p=0.8,
                                         max_tokens=2096)
        self.llm_evaluate_agent = LLMAgent(EVALUATION, self.openai_agent, stream=False)

    def extract_dict_from_json(self,text: str):
        """
        Extract the dictionary between "```json" and "```". 
        """
        pattern = r'```json(.*?)```'
        extracted_text = re.findall(pattern, text, re.DOTALL)

        return extracted_text[0].strip() 

    def flowing(self, prediction: str, true: str) -> str:
        
        result = self.llm_evaluate_agent(prediction=prediction, true=true)
        result = self.extract_dict_from_json(result)

        return result


EVALUATION = [
    {"role" : "system",
     "content": """Now, you are a evaulator who can help user to determine the accurate rate of prediction file compared to the true file.
    Here is a sample of prediction file:
    {{
        "name": "xxx",
        "period_covered": "xxx",
        "address": "xxx",
        "period_covered": "xxx",
        "opening_balance": "xxx",
        "closing_balance": "xxx",
        "loan_amount": "xxx",
    }}
## You must follow all the requirements to complete the task:
    1. You must compare each items that exists with valid value on prediction file to the true file. 
    2. If there are both first name and last name in true file, you need to combine them together as name. If there are several address in true file, you need to combine them together as address.
    3. If the item with valid value exists in the prediction file but not in the true file, it counts as incorrectly predicted.
    4. You must record the items that are correctly predicted.
    5. You must record the items that are incorrectly predicted.
    6. You must count the number of items that are correctly predicted.
    7. You must count the number of items that are incorrectly predicted.
    8. You must calculate the accuracy of the prediction. 

## About the output:
    Your output must be a json file containing a python dictionary to store the result, the format follows this:
    {{
        "correctly_predicted_items": ["xxx", "xxx", "xxx"],
        "incorrectly_predicted_items": ["xxx", "xxx", "xxx"],
        "correctly_predicted": "xxx",
        "incorrectly_predicted": "xxx",
        "accuracy": "xxx",
    }}
    You must follow all requirements listed above. 
    Your output must contain the json file quoted by "```json" and "```"

    """},
    {"role": "user",
    "content": """
        Here is the prediction file: {prediction}. 
        Here is the true file: {true}.
"""}]
