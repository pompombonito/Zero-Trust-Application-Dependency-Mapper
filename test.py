# Author: Jairo Devon A. Daquioag
# "Zero Trust Map"
# July 28, 2026
# This is an LLM Assisted Algorithm

"""
Madani, O. (2022). Cisco Secure Workload Networks of Computing Hosts [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51K7Q.
"""

import gzip
import pandas as pd
import networkx as nx
from community import community_louvain
from pyvis.network import Network

def load_cisco_workload_data(file_path, max_lines=None):
    print(f"1. Parsing Cisco dataset: {file_path}...")
    parsed_edges = []
    line_count = 0
    
    with gzip.open(file_path, mode='rt') as file:
        for line in file:
            if line.startswith('#'): continue
            if max_lines and line_count >= max_lines: break
                
            parts = line.strip().split()
            if len(parts) < 4: continue
                
            src_node, dst_node, port_info_csv = parts[1], parts[2], parts[3]
            connections = port_info_csv.split(',')
            
            for conn in connections:
                if 'p' not in conn or '-' not in conn: continue
                try:
                    port_proto, packets = conn.split('-')
                    port, protocol = port_proto.split('p')
                    parsed_edges.append({
                        'src_node': src_node,
                        'dst_node': dst_node,
                        'port': int(port),
                        'protocol': int(protocol),
                        'packets': int(packets) # This is the traffic weight!
                    })
                except ValueError:
                    continue
            line_count += 1
            
    return pd.DataFrame(parsed_edges)

# --- THE PIPELINE ---

# 1. Load Data (Limiting to 5000 lines so the web map renders quickly)--
# actually, I need to optimize this further as it still loads like a turtle
df = load_cisco_workload_data("out1_1.txt.gz", max_lines=5000)

# 2. Build Graph
print("2. Building the NetworkX Graph...")
G = nx.from_pandas_edgelist(
    df, source="src_node", target="dst_node", edge_attr=["port", "protocol", "packets"]
)

# 3. Machine Learning Clustering
print("3. Running Louvain Community Detection...")
# Unlike my supervised Scikit-Learn classifiers I am evaluating in my thesis, 
# this algorithm is completely unsupervised. It finds the patterns blindly.
partition = community_louvain.best_partition(G)

# Color the nodes based on their AI-assigned group
colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#FF33F3', '#33FFF3']
for node, group_id in partition.items():
    G.nodes[node]['group'] = group_id
    G.nodes[node]['title'] = f"Node IP ID: {node}\nApp Group: {group_id}"
    G.nodes[node]['color'] = colors[group_id % len(colors)]

# 4. Generate Visualization
print("4. Generating Interactive Map...")
net = Network(notebook=False, height="750px", width="100%", bgcolor="#222222", font_color="white")
net.from_nx(G)
net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)
net.show("cisco_zero_trust_map.html", notebook=False)

print("\nSUCCESS! Open 'cisco_zero_trust_map.html' in your browser.")