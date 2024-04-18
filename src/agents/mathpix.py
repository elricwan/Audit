"""
=======
mathpix
=======
@date: 2024-3-22
"""
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
print(sys.path)

from xyz.node.agent import Agent
import cv2
import json
import base64
import requests
import numpy as np
import os
from dotenv import load_dotenv


class MathpixProcessor(Agent):
    def __init__(self, app_id, app_key, content_type='application/json'):
        """Initiation

        Parameters
        ----------
        app_id : str
            app id of mathpix application
        app_key : str
            api key of mathpix application
        content_type : str, optional
            decide if the image are uploaded in json or url, if by url then should be none, by default 'application/json'
        """
        super().__init__() 

        self.headers = {
            "app_id": app_id,
            "app_key": app_key
        }
        if content_type:
            self.headers["Content-type"] = content_type

        self.set_name("MathpixProcessor")
        self.set_description("This is a function convert image to latex")
        self.set_parameters({"image_path": {"type": "str", "description": "This is the path of image"}})

    def flowing(self, image_path: str) -> tuple[str, list[np.array]]:
        """function for recognizing text in image using mathpix text api and crop out images

        Parameters
        ----------
        image_path : str
            path to the image to be ocr

        Returns
        -------
        tuple[str, list[np.array]]
            str is text in the image, list of np.array is list of images
        """
        #read in images
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Making the request to Mathpix API
        payload = {
            "src": "data:image/jpeg;base64," + image_base64,
            "formats": ["text", "data"],
            "data_options": {
                "include_latex": True
            },
            "include_line_data": True
        }
        
        # Get mathpix response
        response = requests.post("https://api.mathpix.com/v3/text", headers=self.headers, data=json.dumps(payload))
        result = response.json()
    
        if 'error' in result.keys():
            print(result)
            
        text = result['text']
        image_countours = [data['cnt'] for data in result['line_data'] if data['type'] == 'diagram']

        if image_countours:
            # Read in original image and crop out
            image = cv2.imread(image_path)
            image_boxs = [image_contour[1] + image_contour[3] for image_contour in image_countours]
            cropped_images = [image[bbox[1]:bbox[3], bbox[0]:bbox[2]] for bbox in image_boxs]
        else:
            cropped_images = []
            
        return text, cropped_images
