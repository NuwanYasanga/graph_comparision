from networkx.classes.graph import Graph
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def calculate_averages(df):

    averages = df.groupby('Keys')['F3'].mean().reset_index()
    return averages

def average_f3_timing(df, row_id):
    total_timing = df['F3'].sum()
    avg_time = round(((df['F3'][row_id])/total_timing),3)

    return avg_time

def creating_graph(df):
    G = nx.MultiDiGraph()
    weights = []
    for i in range(len(df)):
            key_pair = df['Keys'][i]
            first = key_pair[:key_pair.find(',')]
            second = key_pair[key_pair.find(',')+1:]
            time = average_f3_timing(df,i) * 100
            G.add_edge(first, second,weight = time)
            weights.append(time)
    return G, weights


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