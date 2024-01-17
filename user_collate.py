import os
import pandas as pd

# Set the data folder paths
data_folder = "/scratch3/kal058/Graph_embedding/Data_output/Jaccard_similarity/"
write_folder = "/scratch3/kal058/Graph_embedding/Data_output/Jaccard_similarity"

# Get the list of files in the data folder
files_list = os.listdir(data_folder)
all_users_summary = pd.DataFrame()

# Loop over the files and read each one
for i, file in enumerate(files_list):
    # Construct the full file path
    file_path = os.path.join(data_folder, file)
    # Read the CSV file into a DataFrame
    dat = pd.read_csv(file_path)
 
    # Append the data to the summary DataFrame
    all_users_summary = pd.concat([all_users_summary, dat], ignore_index=True)

# Print the column names
print(len(all_users_summary))

# Write the combined DataFrame to a CSV file
output_file = "all_users_graphs.csv"
output_path = os.path.join(write_folder, output_file)
all_users_summary.to_csv(output_path, index=False)
