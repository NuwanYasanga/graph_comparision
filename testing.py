import networkx as nx

def modified_jaccard_similarity(G1, G2):
    """
    Calculate a modified Jaccard Similarity that considers edge widths.
    """
    # Create sets of (node1, node2, weight) tuples for each graph
    edges1 = {(u, v, G1[u][v]['weight']) for u, v in G1.edges()}
    edges2 = {(u, v, G2[u][v]['weight']) for u, v in G2.edges()}

    # Calculate intersection and union of the edge sets
    intersection = len(edges1.intersection(edges2))
    union = len(edges1.union(edges2))

    # Compute the Jaccard Similarity
    jaccard_similarity = intersection / union if union != 0 else 0
    return edges1, edges2, jaccard_similarity

# Create two graphs with edge widths
G1 = nx.Graph()
G1.add_edge(1, 2, weight=0.5)
G1.add_edge(2, 3, weight=1.0)

G2 = nx.Graph()
G2.add_edge(1, 2, weight=0.5)
G2.add_edge(2, 3, weight=0.75)
G2.add_edge(3, 4, weight=1.0)

# Calculate the modified Jaccard Similarity
ed1, ed2, similarity = modified_jaccard_similarity(G1, G2)
print(ed1)
print(ed2)
print("Modified Jaccard Similarity:", similarity)