from networkx.classes.graph import Graph
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
#from karateclub import DeepWalk
#from karateclub.graph_embedding import Graph2Vec, FGSD, GL2Vec
#from sklearn.decomposition import PCA
#from scipy.stats import zscore
#from sklearn.preprocessing import MinMaxScaler

df_1_sam_1 = pd.read_csv('C:/Research Activities/Datasets/BB-MAS_Dataset/Desktop_data/Desktop_fixed_sentence_samples/User_1@sample_1.csv',header =0)
df_2_sam_1 = pd.read_csv('C:/Research Activities/Datasets/BB-MAS_Dataset/Desktop_data/Desktop_fixed_sentence_samples/User_2@sample_1.csv', header=0)
df_1_sam_3 = pd.read_csv('C:/Research Activities/Datasets/BB-MAS_Dataset/Desktop_data/Desktop_fixed_sentence_samples/User_1@sample_3.csv', header=0)
df_1_sam_9 = pd.read_csv('C:/Research Activities/Datasets/BB-MAS_Dataset/Desktop_data/Desktop_fixed_sentence_samples/User_1@sample_9.csv', header=0)

def calculate_averages(df):

    averages = df.groupby('Keys')['F3'].mean().reset_index()
    return averages

def average_f3_timing(df, row_id):
    total_timing = df['F3'].sum()
    avg_time = round(((df['F3'][row_id])/total_timing),3)

    return avg_time

#def creating_graph(df):
#    G = nx.MultiDiGraph()
#    weights = []
#    for i in range(len(df)):
#            key_pair = df['Keys'][i]
#            first = key_pair[:key_pair.find(',')]
#            second = key_pair[key_pair.find(',')+1:]
#            time = average_f3_timing(df,i) * 100
#            G.add_edge(first, second,weight = time)
#            weights.append(time)
#    return G, weights



def creating_graph(df):
    G = nx.MultiDiGraph()
    weights = []
    for i in range(len(df)):
        key_pair = df['Keys'][i]
        first = key_pair[:key_pair.find(',')]
        second = key_pair[key_pair.find(',')+1:]
        time = round((average_f3_timing(df, i) * 100),2)
        #print(f"Edge ({first}, {second}) with weight: {time}")  # Debug print
        G.add_edge(first.strip(), second.strip(), weight=time)
        weights.append(time)
        
    # Check if any weights are non-zero
    #if any(w == 0 for w in weights):
    #    print("All weights are zero.")
    #else:
    #    print("All weights are non zero.")
    
    return G, weights



def visualise_graph(G, weights, i):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_title(f'Fixed_test_graph_user_'+ str(i))
    #plt.hist([v for k,v in nx.degree(G)])
    pos = nx.spring_layout(G)
    nx.draw(G,pos,node_size=80,font_size=5, with_labels = True, width= weights)
    plt.show()

def modified_jaccard_similarity(G1, G2):
    """
    Calculate a modified Jaccard Similarity that considers edge widths.
    """
    # Create sets of (node1, node2, weight) tuples for each graph
    edges1 = {(u, v, w['weight']) for u, v, w in G1.edges(data=True)}
    edges2 = {(u, v, w['weight']) for u, v, w in G2.edges(data=True)}

    # Calculate intersection and union of the edge sets
    intersection = len(edges1.intersection(edges2))
    union = len(edges1.union(edges2))

    # Compute the Jaccard Similarity
    jaccard_similarity = intersection / union if union != 0 else 0
    return edges1, edges2, jaccard_similarity


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



# Calculate the buffered Jaccard similarity
    jaccard_similarity = modified_intersection_count / len(union) if len(union) > 0 else 0
    return jaccard_similarity


df = calculate_averages(df_1_sam_1)
Graph1, weights1 = creating_graph(df)
#visualise_graph(Graph1, weights1,1)

df = calculate_averages(df_2_sam_1)
Graph2, weights2 = creating_graph(df)
#visualise_graph(Graph2, weights2,2)

df = calculate_averages(df_1_sam_3)
Graph13, weights13 = creating_graph(df)

df = calculate_averages(df_1_sam_9)
Graph19, weights19 = creating_graph(df)

ed1, ed2, similarity1 = modified_jaccard_similarity(Graph1,Graph2)
ed11, ed12, similarity12 = modified_jaccard_similarity(Graph1,Graph19)
ed11, ed13, similarity12 = modified_jaccard_similarity(Graph1,Graph13)


bu_similarity1  = buffered_jaccard_similarity(Graph1,Graph2)
bu_similarity19  = buffered_jaccard_similarity(Graph1,Graph19)

#edges = {tuple(sorted((u, v))) for u, v in Graph1.edges()}
#for edge in edges:
#    print(edge)
#    print(Graph1.get_edge_data(edge[0], edge[1]))

#print(ed1)
#print(ed2)
print(similarity1)
print(bu_similarity1)
#print()
#print(ed11)
#print(ed13)
print(similarity12)
print(bu_similarity19)

#for u, v, weight in Graph1.edges(data=True):
#    print(weight['weight'])


