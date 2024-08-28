# This script performs the same task as the "path_quality_scorer.py", but it is designed to work using OpenAI Batch API

import os, sys, argparse
import pandas as pd
import json
from tqdm import tqdm
import time

from utils.openai_api import OpenAIHandler
from utils.base_functions import str2bool

import tiktoken


def pass_arguments():
    parser = argparse.ArgumentParser(description='Path quality evaluation.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input', type=str, default=None, help='Path to the input batch file.')
    parser.add_argument('--model', type=str, default=None, help='OpenAI model.')
    parser.add_argument('--hop', type=int, default=2, help='Number of hops.')
    parser.add_argument('--monitor', type=str2bool, default=True, help='Monitor the batch status.')
    parser.add_argument('--output_list_file', type=str, default='list_of_outputs.txt', help='Name of the file where the list of outputs will be saved.')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = pass_arguments()
    if '.json' in args.input or '.jsonl' in args.input:
        output_name = args.input.split('.json')[0]
    else:
        raise ValueError('Your input file should be in .json or .jsonl format!')

    # launch the bot
    bot = OpenAIHandler()

    # upload the batch file
    batch_input_file = bot.batch_upload(f'./data/batch_input/{args.input}') 

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
    print('Batch information:')

    for e in info:
        print(f'\t{e}: {info[e]}')
    print()

    # monitor the status of the batch
    if args.monitor:
        print('Monitoring the batch status every 60 seconds...')    
        
        while True:
            tmp = bot.batch_info(info['id'])
            status = bot.batch_info(info['id'])['status']
            
            print(f'\t{status}')

            if status == "completed" or status == "failed" or status == "cancelled" or status == "expired":
                break
            time.sleep(60) # check the status every 60 seconds
    print()

    # Note that the line above will print out the batch ID, which can be used to check the status of the batch
    # for more information please refer to https://platform.openai.com/docs/guides/batch/getting-started

    # in case if the batch is completed, retrieve the results
    completed_batch = bot.batch_info(info['id'])
    if completed_batch['status'] == 'completed':

        # print the results information
        print('Batch was completed successfully!')
        for e in completed_batch:
            print(f'\t{e}: {completed_batch[e]}')
        print()

        # retrieve the output file
        file_response = bot.batch_retrieve(completed_batch['output_file_id'])

        # save the results to a file
        bot.batch_save_results(file_response.text, f'./data/batch_output/{output_name}_result.jsonl') 

        # check if ar.gs.output_list_file exists, if not create it
        if not os.path.exists(f"./data/batch_output/{args.output_list_file}"):
            with open(f'./data/batch_output/{args.output_list_file}', 'w') as f:
                pass

        # save the output file name to the list
        with open(f'./data/batch_output/{args.output_list_file}', 'a', newline='') as f:
            f.write(f'{output_name}_result.jsonl\n')

        # TODO: The error file part should be reworked, there is a better way to handle it
        # Also the code right now doesn't handle the case when the error file is found
        try:
            # retrieve the error file
            error_file = bot.batch_retrieve(completed_batch['error_file_id'])

            # save the error file to a file
            bot.batch_save_results(error_file.text, f'./data/batch_output/{output_name}_error.jsonl')

        except:
            print('No error file was found!')
            print()

    else:
        print('Batch was not completed successfully. Maybe it is still running!')
        print('\tStatus:', completed_batch['status'])
        print()
