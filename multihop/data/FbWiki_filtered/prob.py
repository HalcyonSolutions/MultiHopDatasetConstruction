import pandas as pd
from collections import Counter

'''
df = pd.read_csv('raw.csv',header=None)
fst = df.iloc[:,0].to_list()
scd = df.iloc[:,2].to_list()
'''

'''
fst, scd = [], []
with open('train.triples', 'r') as f:
    for line in f:
        h1, h2, _ = line.strip().split()
        fst.append(h1)
        scd.append(h2)
'''

fst, scd = [], []
with open('raw.kb', 'r') as f:
    for line in f:
        h1, h2, _ = line.strip().split()
        fst.append(h1)
        scd.append(h2)
strings = fst + scd

counts = Counter(strings)

# Total number of strings
total_count = len(strings)

# Calculate the probability of each unique string
probabilities = {string: count / total_count for string, count in counts.items()}

sorted_probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))

# Print the probabilities
with open('raw.pgrk', 'w') as f:
    for string, probability in sorted_probabilities.items():
        f.write(f'{string:<30}:{probability:.14g}\n')

print(f'Number of unique nodes: {len(probabilities.items())}') 
