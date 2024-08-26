import os
from dotenv import load_dotenv # to load the openai keys from the environment variable

from openai import OpenAI
import tiktoken


# dictionary for pricing per 1000 tokens
pricing_input = {
    'gpt-4':0.03,
    'gpt-4-turbo':0.01,
    'gpt-4o-mini':0.00015
}

pricing_output = {
    'gpt-4':0.06,
    'gpt-4-turbo':0.03,
    'gpt-4o-mini':0.0006
}


class OpenAIHandler:
    def __init__(self, model):
        load_dotenv()
        
        # "OPENAI_API_KEY" is the key that is stored in the ".env" file
        # the key is used to access the OpenAI API
        # using OpenAI() client we can access the models
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

        # select either "cl100k_base" or your "model" as the encoding schema
        # self.encoding = tiktoken.get_encoding("cl100k_base")
        self.encoding = tiktoken.encoding_for_model(model)

        print(f"OpenAIHandler initialized with model: {self.model}")

    def query(self, prompt):
        query_info={'input_tokens':0, 'output_tokens':0, 'input_cost':0, 'output_cost':0}

        # count the number of tokens in the input
        # count the cost of the input
        # cost is calculated in USD$
        query_info['input_tokens'] = len(self.encoding.encode(prompt))
        query_info['input_cost'] = query_info['input_tokens']/1000 * pricing_input[self.model]

        # get the response from the model, providing the model name and the prompt
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        answer = (response.choices[0].message.content)  

        # count the number of tokens in the output
        # count the cost of the output
        query_info['output_tokens'] = len(self.encoding.encode(answer))
        query_info['output_cost'] = query_info['output_tokens']/1000 * pricing_output[self.model]

        query_info['answer'] = answer
        
        return query_info
