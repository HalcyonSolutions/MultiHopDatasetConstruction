import os
from dotenv import load_dotenv # to load the openai keys from the environment variable

from openai import OpenAI
import tiktoken

# Below are the dictionaries for pricing per 1000 tokens
# * The pricing is in USD$

# TODO: Verify the pricing information
# * OpenAI says that it's 50% less expensive to use the Batch API

pricing_input = {
    'gpt-4':0.03,
    'gpt-4-turbo':0.01,
    'gpt-4o-mini':0.00015,
    'gpt-4o': 0.005
}

pricing_output = {
    'gpt-4':0.06,
    'gpt-4-turbo':0.03,
    'gpt-4o-mini':0.0006,
    'gpt-4o': 0.015
}

pricing_embeddings = {
    'text-embedding-3-small': 0.020,
    'text-embedding-3-large': 0.130,
    'ada v2': 0.100
    }

class OpenAIHandler:
    def __init__(self, model=None, embedder=None, tokenizer=None, encoding=None):
        # load the environment variables from the ".env" file
        load_dotenv()
       
        # "OPENAI_API_KEY" is the API key stored in the ".env" file.
        # This key is required to authenticate and access the OpenAI API.
        # The OpenAI() client utilizes this key to interact with the available models.
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model # model is the name of the model, used to generate the response
        self.embedder = embedder # embedder is the model, used to generate the response
        self.tokenizer = tokenizer # tokenizer is the tokenizer, used to count the number of tokens in the input and output
        self.encoding = encoding
        
        # print the information about the OpenAIHandler
        if self.model is not None:
            print(f"\tModel: {self.model}")
        if self.embedder is not None:
            print(f"\tEmbedder: {self.embedder}")
        if self.tokenizer is not None:
            print(f"\tTokenizer: {self.tokenizer}")
        if self.encoding is not None:
            print(f"\tEncoder: {self.encoding}")
        print()

    """
    Below are the functions to interact with the Chat API
    """
    # query the model with the prompt
    def query(self, prompt):
        # create the tokenizer if it is not provided
        if self.tokenizer is None:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        
        # create a dictionary to store the information about the query
        query_info = {}
        
        # count the number of tokens in the input
        # count the cost of the input
        # query_info['input_tokens'] = len(self.tokenizer.encode(prompt))
        # query_info['input_cost'] = query_info['input_tokens']/1000 * pricing_input[self.model]

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
        # query_info['output_tokens'] = len(self..encode(answer))
        # query_info['output_cost'] = query_info['output_tokens']/1000 * pricing_output[self.model]

        query_info['answer'] = answer
        
        return query_info

    def get_embedding(self, text) -> list:
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input = [text], model=self.model).data[0].embedding
    
    def encode(self, text) -> list:
        return self.get_embedding(text)
    
    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(self.encoding)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    """
        Below are the functions to interact with the Batch API
    """
    # upload the batch file
    def batch_upload(self, file_path):
        batch_input_file = self.client.files.create(
            file=open(file_path, "rb"),
            purpose="batch"
        )
        return batch_input_file

    # returns information about all batches that you had created   
    def batch_list(self, limit=10):
        return self.client.batches.list(limit=limit)

    # returns information about a specific batch process
    def batch_info(self, batch_id):
        info = self.client.batches.retrieve(batch_id)
        return dict(info) # by default its a tuple, easier to return info from a dictionary

    # retreive the results from the batch process
    def batch_retrieve(self, output_file_id):
        return self.client.files.content(output_file_id)

    # cancel the batch process
    def batch_cancel(self, batch_id):
        return self.client.batches.cancel(batch_id)
    
    # Save the results from the batch api to a .jsonl file
    def batch_save_results(self, file_text, output_file):
        with open(output_file, 'w') as output:
            output.write(file_text)
        print(f'Results are saved into {output_file}', end='\n\n')