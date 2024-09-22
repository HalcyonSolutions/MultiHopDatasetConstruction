# This script performs the same task as the "path_quality_scorer.py", but it is designed to work using OpenAI Batch API

import os, sys, argparse
import time
import json
import tiktoken

from utils.openai_api import OpenAIHandler, pricing_output
from utils.base_functions import str2bool

# import tiktoken


def pass_arguments():
    parser = argparse.ArgumentParser(description='Path quality evaluation.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input_folder', type=str, default=None, help='Path to the input batch file.')
    parser.add_argument('--output_folder', type=str, default='list_of_outputs.txt', help='Name of the file where the list of outputs will be saved.')
    parser.add_argument('--model', type=str, default=None, help='OpenAI model.')
    parser.add_argument('--hop', type=int, default=2, help='Number of hops.')
    parser.add_argument('--monitor', type=str2bool, default=True, help='Monitor the batch status.')

    args = parser.parse_args()

    return args


def run(model, input_folder, input_file, output_folder, output_name, monitor):
    # launch the bot
    bot = OpenAIHandler()

    # upload the batch file
    batch_input_file = bot.batch_upload(f'./data/batch_input/{input_folder}/{input_file}') 

    # print the batch input file ID
    # you can use this ID to check the input file
    # that was uploaded to the OpenAI
    batch_input_file_id = batch_input_file.id
    print(f"Batch input file ID: {batch_input_file_id}")

    # create the batch
    # * not a class method, because there are too many parameters
    info = bot.client.batches.create(
        input_file_id=batch_input_file_id, # the ID of the file that contains the batch
        endpoint="/v1/chat/completions", # the endpoint that the batch will be sent to
        completion_window="24h", # 24 hours, the maximum time allowed for the batch to complete
        metadata={
            "description": "Path quality scorer",
        }
    )
    info = dict(info) # convert from tuple to dictionary
    
    # print the batch information
    print("Batch information:")
    print("\n".join(f"\t{key}: {value}" for key, value in info.items()))
    print()


    # monitor the status of the batch
    if monitor:
        print('Monitoring the batch status every 60 seconds...')    
        
        while True:
            status = bot.batch_info(info['id'])['status']
            
            print(f'\t{status}')

            if status == "completed" or status == "failed" or status == "cancelled" or status == "expired":
                break
            time.sleep(60) # check the status every 60 seconds
        print()


    # in case if the batch is completed, retrieve the results
    completed_batch = bot.batch_info(info['id'])
    if completed_batch['status'] == 'completed':
        output_path = f'data/batch_output/{output_folder}'

        # print the results information
        print("Batch was completed successfully!")
        print("\n".join(f"\t{key}: {value}" for key, value in completed_batch.items()))
        print()
        
        # retrieve the output file
        file_response = bot.batch_retrieve(completed_batch['output_file_id'])

        # file_response is a response object, so we need to get the text from it, it is a jsonl file in string format
        # extract content from each line of the jsonl file
        # using tiktoken extract content from each line of the jsonl file
        # and calculte number of tokens and cost using pricing_output[model]

        total_toekns = 0
        encoding = tiktoken.encoding_for_model(model)
        print('Processing output batch...')

        for line in file_response.text.split('\n'):
            # extract conent from this line
            # convert line into json
            if len(line) == 0:
                continue
            line = json.loads(line)
            content = line['response']['body']['choices'][0]['message']['content']
            total_toekns += len(encoding.encode(content))
        print()
        
        # calculate the cost of the output
        cost = total_toekns/1000 * pricing_output[model] / 2
        print(f'Total tokens generated: {total_toekns}')
        print(f'Total cost: ${cost}')

        # save the results to a file
        bot.batch_save_results(file_response.text, f'{output_path}/{output_name}_result.jsonl') 

        # TODO: The error file part should be reworked, there is a better way to handle it
        # Also the code right now doesn't handle the case when the error file is found
        try:
            # retrieve the error file
            error_file = bot.batch_retrieve(completed_batch['error_file_id'])

            # save the error file to a file
            bot.batch_save_results(error_file.text, f'{output_path}/{output_name}_error.jsonl')

        except:
            print('No error file was found!\n')

    else:
        print('Batch was not completed successfully. Maybe it is still running!')
        print('\tStatus:', completed_batch['status'])
        print()


if __name__ == "__main__":
    args = pass_arguments()

    # read the name of the files in the args.input_folder
    filenames = os.listdir(f'./data/batch_input/{args.input_folder}')
    # read line by line from args.input_folder
    for input_file in filenames:
        print(f'Processing file: {input_file}')
        run(args.model, input_folder = args.input_folder, input_file=input_file, output_folder=args.output_folder, output_name=f"{input_file.split('.jsonl')[0]}", monitor=args.monitor)
