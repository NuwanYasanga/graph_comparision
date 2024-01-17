import pandas as pd
import os
import pickle

def buffered_jaccard_similarity(G1, G2, buffer=0.1):
   
    edges1 = {(u, v) for u, v in G1.edges()}
    edges2 = {(u, v) for u, v in G2.edges()}
    
    # Intersection and union for calculation
    intersection = edges1.intersection(edges2)
    union = edges1.union(edges2)

    modified_intersection_count = 0
    for edge in intersection:
        weight1 = G1.get_edge_data(edge[0],edge[1])[0]['weight']
        weight2 = G2.get_edge_data(edge[0],edge[1])[0]['weight']

        weight_diff = round(abs(weight1 - weight2),2)


        if weight_diff <= buffer:
            modified_intersection_count += 1

    jaccard_similarity = modified_intersection_count / len(union) if len(union) > 0 else 0
    return jaccard_similarity

data_dir = 'C:/Research Activities/Datasets/BB-MAS_Dataset/Desktop_data/Desktop_fixed_sentence_samples/'

user_files_desktop = os.listdir(data_dir)

user_id = 1 

user1_list = []
user1_sample_list = []
user2_list = []
user2_sample_list = []
score_list = []

for i in range(len(user_files_desktop)):
    user1_file = user_files_desktop[i]
    user1_id = int(user1_file[user1_file.find('_')+1:user1_file.find('@')])
    if user1_id == user_id:
        user1_sample = int(user1_file[user1_file.rfind('_')+1:user1_file.find('.')])
        if user1_sample % 2 != 0 and 0 <= user1_sample <= 20:
            with open(f'all_users/User_'+str(user1_id)+'graph_'+str(user1_sample)+'.pickle', 'rb') as f:
                G1 = pickle.load(f)
        
            
        
            for j in range(0,5):
                user2_file = user_files_desktop[j]
                user2_id = int(user2_file[user2_file.find('_')+1:user2_file.find('@')])
                user2_sample = int(user2_file[user2_file.rfind('_')+1:user2_file.find('.')])
                if user2_sample % 2 != 0 and 0 <= user2_sample <= 20:
                    with open(f'all_users/User_'+str(user2_id)+'graph_'+str(user2_sample)+'.pickle', 'rb') as f:
                        G2 = pickle.load(f)

                    similarity = buffered_jaccard_similarity(G1, G2)

                    user1_list.append(user1_id)
                    user1_sample_list.append(user1_sample)
                    user2_list.append(user2_id)
                    user2_sample_list.append(user2_sample)
                    score_list.append(similarity) 

df = pd.DataFrame(list(zip(user1_list,user1_sample_list,user2_list,user2_sample_list, score_list)), columns = ['user1','Sample1','user2','Sample2','score'])
print(df)

#file_name = f'Data_outputs/jaccard_similarity/User_'+str(user_id)+'graph_score.csv'
#df.to_csv(file_name)
