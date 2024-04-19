import os
import time
import traceback
import json
from typing import Dict
import re
import time

from agents.dbrx_client import dbrx_GradioAPIClient
from agents.llama_client import llama2_GradioAPIClient


# api_url = "https://16c187ae810a38fd56-dbrx-instruct.test-playground-inference.netmind.ai/"
# client = dbrx_GradioAPIClient(api_url)
api_url = "https://2c76d788cbff087d88-llama2-70b.test-playground-inference-2.netmind.ai/"
client = llama2_GradioAPIClient(api_url)
request = """Now, you are a Audit assistant who can help user to extract information from text.
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

"""

def extract_dict_from_json(text: str):
    """
    Extract the dictionary between "```json" and "```". 
    """
    pattern = r'```json(.*?)```'
    extracted_text = re.findall(pattern, text, re.DOTALL)

    return extracted_text[0].strip() 

with open('json/latex_info.json', 'r') as f:
    latex_info = json.load(f)

extract_llama = {}
idx = 0
for key in latex_info:
    client = llama2_GradioAPIClient(api_url)
    idx += 1
    print(idx)
    status = 0
    time.sleep(20)
    while status == 0:
        try:
            response = client.run(
                message="The given text is:   \n" + latex_info[key],
                request_description=request
            )
            ans = response['content']
            extract_llama[key] = ans
            status = 1
        except:
            time.sleep(10)
            status = 0

with open('json/extract_llama2.json', 'w') as f:
    json.dump(extract_llama, f)