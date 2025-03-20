"""
@author: Eduin Hernandez

Summary: This script reads a gzipped QID file containing Freebase MID and Wikidata QID
 mappings, processes each line, and converts the data into a CSV format.
 It uses multithreading to speed up processing, with error handling to log problematic
 lines separately. It assumes you have fb2w.nt.gz

Next Code: freebase_2_wikidata - part II

TODO: Clean up code
"""


import csv
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

gzip_file_path = 'fb2w.nt.gz'
output_csv_path = 'mid_qid.csv'
error_log_path = 'mid_qid_error_log.csv'

def process_line(line):
    try:
        parts = line.strip().split()
        if len(parts) < 3:
            raise ValueError("Line does not contain enough parts")

        # Process the MID
        mid_raw = parts[0]
        mid_transformed = '/' + mid_raw.replace('<http://rdf.freebase.com/ns/', '').replace('>', '').replace('.', '/')

        # Extract the encoded title
        title_raw = parts[2]
        encoded_title = title_raw.replace('<http://www.wikidata.org/entity/', '').replace('>', '')

        return (mid_transformed, encoded_title), None
    except Exception as e:
        return None, (mid_raw, str(e))

def main():
    with gzip.open(gzip_file_path, 'rt') as gz_file, \
         open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file, \
         open(error_log_path, 'w', newline='', encoding='utf-8') as error_file:
        
        csv_writer = csv.writer(csv_file)
        error_writer = csv.writer(error_file)
        csv_writer.writerow(['MID', 'QID'])  # Write header to the main CSV
        error_writer.writerow(['MID', 'Error'])  # Write header to the error log CSV
        
        lines = gz_file.readlines()  # Read all lines at once
        futures = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            for line in lines:
                futures.append(executor.submit(process_line, line))

            # Implement tqdm in the as_completed loop
            for future in tqdm(as_completed(futures), total=len(lines), desc="Processing Lines"):
                result, error = future.result()
                if result:
                    csv_writer.writerow(result)
                if error:
                    error_writer.writerow(error)

    print("Conversion complete. Data written to", output_csv_path)
    print("Error log written to", error_log_path)

if __name__ == "__main__":
    main()
