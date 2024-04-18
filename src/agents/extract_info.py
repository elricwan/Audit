"""
=======
pdf extractor
=======
@date: 2024-3-28
"""

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
print(sys.path)

from dotenv import load_dotenv
from xyz.node.agent import Agent
from xyz.node.basic.llm_agent import LLMAgent
from xyz.utils.llm.openai_client import OpenAIClient
import os
import re
from agents.mathpix import MathpixProcessor
# Load the environment variables from the .env file
load_dotenv()
app_key = os.getenv('mathpix_app_key')
app_id = os.getenv('mathpix_app_id')
mathpix_processor = MathpixProcessor(app_id=app_id, app_key=app_key)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


class Extract_info(Agent):
    def __init__(self):
        self.openai_agent = OpenAIClient(api_key=OPENAI_API_KEY, model='gpt-4-0125-preview', temperature=0., top_p=1.0,
                                        max_tokens=2096)
        super().__init__()

        self.llm_cot_agent = LLMAgent(AUDIT, self.openai_agent, stream=False)

    def img2latex(self,image_path):
        """Input: img Filepath, Output: latex string."""
        text, _ = mathpix_processor(image_path=image_path)
        return text 
    
    def extract_dict_from_json(self,text: str):
        """
        Extract the dictionary between "```json" and "```". 
        """
        pattern = r'```json(.*?)```'
        extracted_text = re.findall(pattern, text, re.DOTALL)

        return extracted_text[0].strip()
    
    def flowing(self, text: str) -> str:
        response = self.llm_cot_agent(text=text)
        response = self.extract_dict_from_json(response)

        return response


AUDIT = [
    {"role" : "system",
     "content": """Now, you are a Audit assistant who can help user to extract information from text.
    ## You must follow all the requirements to modify the draft:
        1. You must extract the name of person from the text, including first and last name.
        2. You must extract the period_covered from the text, if given.
        3. You must extract the address from the text, if given.
        4. You must extract the Opening Balance from the text, if given.
        5. You must extract the Closing Balance from the text only if given.
        6. You must extract the loan amount from the text only if the text is about loan application.
    
    ## About the output:
        Your output must be a json file containing a python dictionary to store the extracted information in the format looks like this: 
        
        {{
            "name": "xxx",
            "period_covered": "xxx",
            "address": "xxx",
            "period_covered": "xxx",
            "opening_balance": "xxx",
            "closing_balance": "xxx",
            "loan_amount": "xxx",
        }}
        You must follow all requirements listed above. 
        Your output must contain the json file quoted by "```json" and "```"

    """},
    {"role": "user",
    "content": """
        The given text is:

    {text}
"""}]