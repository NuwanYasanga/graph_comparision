from networkx.classes import graph
import pandas as pd
import os
from jacccard_similarity import calculate_averages, creating_graph, buffered_jaccard_similarity
import pickle


data_dir = 'C:/Research Activities/Datasets/BB-MAS_Dataset/Desktop_data/Desktop_fixed_sentence_samples/'

user_files_desktop = os.listdir(data_dir)

graphs = []
user_list = []
sample_list = []



for i in range(len(user_files_desktop)):
    user_file = user_files_desktop[i]
    current_user_id = int(user_file[user_file.find('_')+1:user_file.find('@')])
    curr_user_sample = int(user_file[user_file.rfind('_')+1:user_file.find('.')])
    df = pd.read_csv(data_dir+user_file, header=0)

    if len(df)>2 : 
        df_avg = calculate_averages(df)
        G, _ = creating_graph(df_avg)

    with open(f'all_users/User_'+str(current_user_id)+'graph_'+str(curr_user_sample)+'.pickle', 'wb') as f:
        pickle.dump(G, f)


