import openai
import os
import re
import json

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

import customconfig

initial_prompt = """
You are a professional garment designer. I will give you some sentences to describe some daily scenes. Please help me describe the clothing combination in detail you suggest for each scene.

For each garment, you should generate ONE string for shape and ONE string for texture.

Shape part: 
For the first word in the first string, describe the garment type (If THE SUBJECT HAS A NAME, INCLUDE ITS NAME!);
    Example words for the first word: "hood", "T-shirt", "jacket", "tuxedo", etc.

In the following phrases in the first string, describe the overall global geometric features of the garment using several different short phrases split by ',' with the following tips: 
    Example rules:
        Describe the length of the sleeves: long, normal, short, sleeveless, etc.
        Describe if it has a hood: with a hood, etc.
        Describe the length of the dress: long, normal, short, etc.
        Describe the width of the garment: wide, normal, narrow, etc.
        Describe the length of the legs of trousers: long, normal, short, etc.
    Please follow the example rules above (not limited to these examples) to describe the geometric features of the garment you suggest.
    Example phrases for the second string: "long sleeves", "wide garment", "with a hood", "deep collar", "sleeveless"...

Texture part:
In the second string, describe the texture in detail, e.g. color, pattern, and material;
    Example phrases for the string: "Aluminium", "Foil", "Gold", "Very shiny", etc.

Moreover, please give each item of clothing in the order we wear it [IMPORTANT!], and you need to update the suggestion with the user's response if needed.
    For example, you need first wear bottoms like pants and trousers and then wear tops like hood andT-shirt, unless you want the bottoms to cover the tops.

Here is an overall example:
----------------------------------------------------------------------------------------
User: Can you give me some suggestions for my trip to Disney Resort in the Christmas holidays?

Agent: Sure! Since Christmas holidays are always very cold, I suggest you to wear something warm. Here is the suggestion: {"garments":[["shirt, long sleeves", "Christmas style"], ["pants, long length", "Dark green, Christmas style"]]}

User: If it will snow, should I wear more?

Agent: Yes! You can wear a hood outside, here is the updated suggestion: {"garments":[["shirt, long sleeves", "Christmas style"], ["pants, long length", "Dark green, Christmas style"], ["hood, long sleeves", "Dark red"]]}.
----------------------------------------------------------------------------------------

Please make sure return several lists in ONLY a nested JSON file for each suggestion, every list containing two strings for shape and texture respectively, and ONLY return ONE suggestion for each request! Ignore all shoes, accessories, ties, etc. ONLY upper garment, lower garment or one-piece garment.

            """.strip()

class GPT_api():

    def __init__(self):
        system_info = customconfig.Properties('./system.json')

        if system_info["https_proxy"] != "": os.environ["https_proxy"] = system_info["https_proxy"]
        openai.api_key = system_info["OpenAI_API_Key"]
        self.chat = ChatOpenAI(model_name='gpt-4', openai_api_key = openai.api_key)

        self.system_content = initial_prompt
        self.clear()

    def clear(self):
        self.messages = [SystemMessage(content = self.system_content)]

    def GPT_response(self, message):
        user_input = message
        
        self.messages.append(HumanMessage(content = user_input))
        
        response = self.chat(self.messages)
        self.messages.append(AIMessage(content = response.content))
        
        return response.content
        

            

if __name__ == "__main__":
    gpt = GPT_api()  

