# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:15:57 2024

@author: Eduin Hernandez

Summary: Webscrapes Wikidata for the information for each entity. 
            Uses threads to speed up information retrieval.
"""
from utils.wikidata import process_entity_data

if __name__ == '__main__':
    input_set_path = './data/modified_fbwiki_nodes.txt'
    output_csv_path = './data/rdf_data.csv'
    
    process_entity_data(input_set_path, output_csv_path, max_workers=15)

#------------------------------------------------------------------------------
# ''' Without Threads '''
# import pandas as pd
# from tqdm import tqdm

# from utils.wikidata import fetch_base_details

# def process_data(file_path, output_file_path):
#     df = pd.read_csv(file_path)

#     results = {
#         'Title': '',
#         'Description': '',
#         'MDI': '',
#         'URL': '',
#         'Alias': '',
#     }

#     titles = []
#     descriptions = []
#     mdi = []
#     wikipedia_url = []
#     aliases = []
    
#     for i0, row in tqdm(df.iterrows(), total=df.shape[0], desc="Fetching data"):
#         if row['RDF']:
#             res = fetch_base_details(row['RDF'], results)
#         else:
#             res = results.copy()  # Handle cases where RDF is blank
            
#         titles.append(res['Title'])
#         descriptions.append(res['Description'])
#         mdi.append(res['MDI'])
#         wikipedia_url.append(res['URL'])
#         aliases.append(res['Alias'])

#     # Add the fetched titles and descriptions to the DataFrame
#     df['Title'] = titles
#     df['Description'] = descriptions
#     df['MDI'] = mdi
#     df['URL'] = wikipedia_url
#     df['Alias'] = aliases
    
#     # Save the updated DataFrame
#     df.to_csv(output_file_path, index=False)
#     print("Data processed and saved to", output_file_path)

# if __name__ == '__main__':
#     input_csv_path = './data/rdf_valid.csv'
#     output_csv_path = './data/rdf_info.csv'
    
#     process_data(input_csv_path, output_csv_path)