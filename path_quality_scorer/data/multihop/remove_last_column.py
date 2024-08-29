import pandas as pd

# Read the CSV file
file_path = '2_hop_filt_10.csv'
df = pd.read_csv(file_path)

# Remove the last column
df = df.iloc[:, :-1]

# Save the modified DataFrame back to the same file
df.to_csv(file_path, index=False)

print(f"File '{file_path}' has been updated by removing the last column.")

