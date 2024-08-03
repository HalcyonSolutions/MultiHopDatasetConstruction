import csv
import argparse
from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
import gc
import sys
import re

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

def classify_relationship(head, relationships):
    # TODO: delete this old prompt if not needed
    #prompt = f"Does the relationship '{relationship}' belong to '{head}'? Please respond with 'yes' or 'no'."
    
    prompt = f"""I will provide a single head, and a list of relationships. Your goal is to assign 0 and 1 to each of the relationship if it belongs to head. Assign 1 if a relationship belongs to a head, if it does not belong to it assign 0. Treat each relationship in the list independently. Result should be a list relationships with their corresponding score (0 or 1), separated by a , (comma) delimiter that follows the relationships in order as I provided. For example: Universe --> country 0, place of birth 1, place of death 1, spouse 0. I provide the head: {head}. I provide the list of 147 relationships: {relationships}, each relationship is separated by a delimiter , (comma)"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    answer = (response.choices[0].message.content)
    return answer

def main(input_csv, relationships):
    df = pd.read_csv(input_csv)
    cols = df.columns[1:]
    rows = df.iloc[:, 0].tolist()
    
    for i, head in enumerate(rows[151:]):
        df = pd.read_csv(input_csv)
        print(f'Row number {i+151}, Head: {head}')

        do = False
        f_count = 0
        while do == False and f_count < 3:
            try:
                classification = classify_relationship(re.sub(r'[^a-zA-Z0-9\s]', '', head), relationships)
                classification = classification.split(',')
                
                digits = []
                for c in classification:
                    digit_str = ''.join(filter(str.isdigit, c))
                    digits.append(int(digit_str))
                df.loc[df['Head'] == head, relationships.split(', ')] = digits

                df.to_csv('output.csv', index=False)
                print('\tFile output.csv is saved!\n')
                do = True

            except:
                print(f'\tFailed for head {head}, trying again')
                f_count += 1
        if f_count == 3:
            with open('failed_heads.txt', 'a') as file:
                file.write('\n' + head)
            print(f'\tFailed to process the head {head}! Wrote the info in the failed_heads.txt!\n')
        del df
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify relationships in a CSV file.')
    parser.add_argument('--input', type=str, help='Path to the input CSV file')
    
    relationships = """country, place of birth, place of death, sex or gender, spouse, country of citizenship, continent, instance of, head of state, capital, official language, currency, position held, shares border with, author, member of sports team, director, screenwriter, discoverer or inventor, ancestral home, educated at, field of work, member of political party, occupation, employer, founded by, home venue, place of burial, basic form of government, publisher, owned by, located in the administrative territorial entity, genre, named after, religion or worldview, based on, contains the administrative territorial entity, follows, headquarters location, cast member, producer, award received, chief executive officer, creator, parent taxon, ethnic group, performer, manufacturer, legislative body, record label, production company, location, programmed in, subclass of, operating system, director of photography, part of, original language of film or TV show, has use, platform, language of work or name, position played on team / speciality, located in time zone, occupational field, distribution format, original broadcaster, unmarried partner, industry, said to be the same as, opposite of, color, member of, chairperson, country of origin, cause of death, honorific prefix, officially opened by, residence, conflict, highest point, religious order, sport, characters, influenced by, location of formation, parent organization, distributed by, symptoms and signs, significant event, authority, notable work, student, mascot, narrative location, filming location, main subject, applies to jurisdiction, conferred by, film editor, location of creation, facet of, instrument, described by source, participant in, winner, replaces, partially coincident with, contains settlement, nominated for, languages spoken, written or signed, affiliation, start point, heritage designation, legal form, has effect, motto, has characteristic, this taxon is source of, health specialty, medical condition treated, history of topic, official symbol, uses, indigenous to, has list, office held by head of the organization, set in period, costume designer, operating area, studied in, subject has role, language used, package management system, film crew member, associated hazard, significant person, sibling, has goal, next higher rank, diaspora, first appearance, risk factor, model item, copyright status, parent, taxon range"""

    args = parser.parse_args()
    
    main(args.input, relationships)
