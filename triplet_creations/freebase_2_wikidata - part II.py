"""

@author: Eduin Hernandez

Summary: Takes MDI from FB15K-237 and converts it to RDF value for WikiData usage.

Previous Code: freebase_2_wikidata - part I
Next Code: fbwiki_triplet_creation

TODO: Clean up code
"""

import pandas as pd

from utils.basic import sort_by_qid

if __name__ == "__main__":
    # Load the node mappings from node_mappings.csv
    node_mappings = pd.read_csv('./data/node_mappings.csv') # manually extract this from the FB15-237 dataset
    
    # Load the output data from output.csv, assuming it contains columns 'MDI' and 'Encoded Title'
    output = pd.read_csv('./data/mdi_rdf.csv')
    
    # Merge the dataframes on the MID column, using a left join to keep all entries from node_mappings
    merged_df = pd.merge(node_mappings, output, on='MDI', how='left')
    
    node_list = merged_df[merged_df['RDF'].notna()]['RDF'].to_list()
            
    # Convert the list to a DataFrame
    node_df = pd.DataFrame(node_list, columns=['Node'])
    
    node_df = sort_by_qid(node_df, column_name = 'Node')
    
    # Save the DataFrame to a txt file, with each node on a new line
    node_df.to_csv('./data/nodes_fb15k.txt', index=False, header=False)
