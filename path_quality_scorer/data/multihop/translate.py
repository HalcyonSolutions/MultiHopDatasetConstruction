# wrtie a code using arpgrapse that takes argument --input a csv file
# the script should do the following
# 1. read the csv file, write it in the df variable using pandas
# 2. create a variable rdfs, and load ../triplet_creations/data/rdf_data.csv
# 3. create a variable props, and load '../triplet_creations/data/relation_data.csv
# 4. iterate through df row by row
#     4.1 iterating thourgh elements in the row
#     4.2 every even element in the row is a RDF, and every odd element is a property
# 5. Using the RDF and property, find the title of the RDF and the title of the property


import argparse
import pandas as pd

RDF_COLUMN_WIDTH = 30
PROP_COLUMN_WIDTH = 20

def main():
    parser = argparse.ArgumentParser(description="Read a CSV file and extract RDF and Property")
    parser.add_argument("--input", type=str, help="The input CSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    rdfs = pd.read_csv('../../../triplet_creations/data/rdf_data.csv')
    props = pd.read_csv('../../../triplet_creations/data/relation_data.csv')

    # get last column names "evaluation_score" from the df and save it in the variable, remove from the df
    last_column = df[df.columns[-1]]
    df = df.drop(columns=[df.columns[-1]])

    for index, row in df.iterrows():
        row_output = []
        for i in range(0, len(row), 2):
            rdf = row[i]
            if i != len(row) - 1:
                prop = row[i + 1]

            # Get titles from lookup dataframes
            rdf_title = rdfs.loc[rdfs['RDF'] == rdf, 'Title'].item()
            if i != len(row) - 1:
                prop_title = props.loc[props['Property'] == prop, 'Title'].item()

            # Format RDF and Property with specific widths for alignment
            rdf_formatted = f"{rdf_title:<{RDF_COLUMN_WIDTH}}"
            if i != len(row) - 1:
                prop_formatted = f"--> {prop_title:<{PROP_COLUMN_WIDTH}}"
                row_output.append(f"{rdf_formatted} {prop_formatted}")
            else:
                row_output.append(f"{rdf_formatted}")

        # Join the formatted row parts and add Quality Score at the end
        formatted_row = ' '.join(row_output)
        quality_score = last_column[index]
        print(f"{formatted_row} | Quality Score: {quality_score:.1f}")
        print()

if __name__ == "__main__":
    main()