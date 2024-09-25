import networkx as nx

def generate_network_report(output_file):
    # Define the graph
    G = nx.Graph()
    
    # Add nodes and edges (replace with your actual graph data)
    edges = [
        (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 6),
        (3, 4), (3, 7),
        (4, 5),
        (5, 6), (5, 8),
        (6, 7),
        (7, 8), (7, 9),
        (8, 9), (8, 10),
        (9, 10)
    ]
    G.add_edges_from(edges)
    
    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # Calculate Freeman degree
    max_degree = max(degree_centrality.values())
    freeman_degree = sum(max_degree - degree for degree in degree_centrality.values()) / ((len(G) - 1) * (len(G) - 2))
    
    # Calculate network density
    density = nx.density(G)
    
    # Generate the report
    with open(output_file, 'w') as f:
        f.write(f"Freeman degree of the network: {freeman_degree:.3f}\n")
        f.write(f"Density of the network: {density:.3f}\n\n")
        
        f.write("Centrality Scores of the Nodes\n")
        f.write("Degree centrality\t\t\tConnectedness centrality\t\t\tBetweenness centrality\n")
        f.write("(From highest to lowest)\t\t(From highest to lowest)\t\t(From highest to lowest)\n")
        
        degree_sorted = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        closeness_sorted = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)
        betweenness_sorted = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
        
        for i in range(len(G)):
            degree_node, degree_score = degree_sorted[i] if i < len(degree_sorted) else ('', '')
            closeness_node, closeness_score = closeness_sorted[i] if i < len(closeness_sorted) else ('', '')
            betweenness_node, betweenness_score = betweenness_sorted[i] if i < len(betweenness_sorted) else ('', '')
            
            f.write(f"Node {degree_node}\t\t\t{degree_score:.3f}\t\t\t")
            f.write(f"Node {closeness_node}\t\t\t{closeness_score:.3f}\t\t\t")
            f.write(f"Node {betweenness_node}\t\t\t{betweenness_score:.3f}\n")

if __name__ == "__main__":
    generate_network_report('network_report.txt')
