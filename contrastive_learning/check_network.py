import os
import re
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import gc  # Garbage collector
import pandas as pd

def parse_dataset_name(folder_name):
    """Parse dataset name to extract information."""
    pattern1 = r'De-noised_(\d+)G_(\d+)T_(\d+)cPerT_dynamics_(\d+)_DS(\d+)'
    pattern2 = r'De-noised_(\d+)G_(\d+)T_(\d+)cPerT_(\d+)_DS(\d+)'
    match_p1 = re.match(pattern1, folder_name)
    match_p2 = re.match(pattern2, folder_name)
    
    if match_p1:
        return {
            'number_genes': int(match_p1.group(1)),
            'number_bins': int(match_p1.group(2)),
            'cells_per_type': int(match_p1.group(3)),
            'dynamics': int(match_p1.group(4)),
            'dataset_id': int(match_p1.group(5)),
            'folder_name': folder_name
        }
    if match_p2:
        return {
            'number_genes': int(match_p2.group(1)),
            'number_bins': int(match_p2.group(2)),
            'cells_per_type': int(match_p2.group(3)),
            'dynamics': int(match_p2.group(4)),
            'dataset_id': int(match_p2.group(5)),
            'folder_name': folder_name
        }
    return None

def get_datasets():
    """Get all available datasets information."""
    datasets = []
    data_sets_dir = '../SERGIO/data_sets'
    
    # Get SERGIO datasets
    if os.path.exists(data_sets_dir):
        for folder_name in os.listdir(data_sets_dir):
            dataset_info = parse_dataset_name(folder_name)
            if dataset_info:
                datasets.append(dataset_info)
    
    # Include new datasets
    new_datasets = [
        {
            'dataset_id': 1001,
            'dataset_name': 'mESC',
            'expression_file': 'data/raws/mESC-ExpressionData.csv',
            'network_file': 'data/raws/mESC-network.csv',
        },
        {
            'dataset_id': 1002,
            'dataset_name': 'mHSC-E',
            'expression_file': 'data/raws/mHSC-E-ExpressionData.csv',
            'network_file': 'data/raws/mHSC-E-network.csv',
        },
        {
            'dataset_id': 1003,
            'dataset_name': 'mHSC-GM',
            'expression_file': 'data/raws/mHSC-GM-ExpressionData.csv',
            'network_file': 'data/raws/mHSC-GM-network.csv',
        },
        {
            'dataset_id': 1004,
            'dataset_name': 'mHSC-L',
            'expression_file': 'data/raws/mHSC-L-ExpressionData.csv',
            'network_file': 'data/raws/mHSC-L-network.csv',
        },
        {
            'dataset_id': 1005,
            'dataset_name': 'mESC-200',
            'expression_file': 'data/raws/mESC-200-ExpressionData.csv',
            'network_file': 'data/raws/mESC-200-network.csv',
        },
        {
            'dataset_id': 1006,
            'dataset_name': 'mHSC-E-200',
            'expression_file': 'data/raws/mHSC-E-200-ExpressionData.csv',
            'network_file': 'data/raws/mHSC-E-200-network.csv',
        },
    ]
    datasets.extend(new_datasets)
    return datasets

class GRNEmbbeddingCycleDetector:
    def __init__(self, dataset_id: int, embeddings: np.ndarray, threshold: float = 0.5, 
                 max_cycle_length: int = 5, max_cycles: int = 1000):
        """Initialize cycle detector using embeddings with memory controls."""
        self.dataset_id = dataset_id
        self.embeddings = embeddings
        self.threshold = threshold
        self.num_genes = embeddings.shape[0]
        self.max_cycle_length = max_cycle_length
        self.max_cycles = max_cycles
        self.graph = None
        
    def _build_graph_from_embeddings(self) -> Dict[int, List[int]]:
        """Convert embeddings to adjacency list representation efficiently."""
        graph = defaultdict(list)
        batch_size = min(1000, self.num_genes)
        
        for i in range(0, self.num_genes, batch_size):
            end_idx = min(i + batch_size, self.num_genes)
            batch_similarities = np.dot(self.embeddings[i:end_idx], self.embeddings.T)
            batch_similarities = (batch_similarities + 1) / 2
            
            for batch_idx, row in enumerate(batch_similarities):
                global_idx = i + batch_idx
                edges = np.where((row > self.threshold) & 
                               (np.arange(self.num_genes) != global_idx))[0]
                graph[global_idx].extend(edges.tolist())
            
            del batch_similarities
            gc.collect()
            
        return graph
    
    def find_cycles(self) -> List[List[int]]:
        """Find cycles efficiently using depth-first search."""
        if self.graph is None:
            self.graph = self._build_graph_from_embeddings()
            
        cycles = []
        visited = set()
        
        def dfs(node: int, path: List[int], start: int, depth: int = 0):
            if len(cycles) >= self.max_cycles or depth >= self.max_cycle_length:
                return
                
            if node in path[1:]:
                cycle = path[path.index(node):]
                if cycle not in cycles:
                    cycles.append(cycle)
                return
                
            for neighbor in self.graph[node]:
                if neighbor == start and depth > 0:
                    new_cycle = path + [neighbor]
                    if new_cycle not in cycles:
                        cycles.append(new_cycle)
                elif neighbor not in path:
                    dfs(neighbor, path + [neighbor], start, depth + 1)
        
        for node in range(self.num_genes):
            if node not in visited and len(cycles) < self.max_cycles:
                dfs(node, [node], node)
                visited.add(node)
        
        return cycles
    
    def get_cycle_stats(self) -> Dict:
        """Get cycle statistics with memory constraints."""
        cycles = self.find_cycles()
        
        if not cycles:
            return {
                'total_cycles': 0,
                'cycles': [],
                'genes_in_cycles': 0,
                'cycle_lengths': [],
                'avg_cycle_length': 0,
                'max_cycle_length': 0,
                'min_cycle_length': 0
            }
        
        genes_in_cycles = set()
        for cycle in cycles:
            genes_in_cycles.update(cycle)
            
        return {
            'total_cycles': len(cycles),
            'cycles': cycles[:10],
            'genes_in_cycles': len(genes_in_cycles),
            'cycle_lengths': [len(cycle) for cycle in cycles],
            'avg_cycle_length': np.mean([len(cycle) for cycle in cycles]),
            'max_cycle_length': max(len(cycle) for cycle in cycles),
            'min_cycle_length': min(len(cycle) for cycle in cycles)
        }
    
    def visualize_cycles(self, save_path: str = None, max_nodes: int = 100):
        """Visualize detected cycles with size limits."""
        if self.graph is None:
            self.graph = self._build_graph_from_embeddings()
            
        cycles = self.find_cycles()
        if not cycles:
            print("No cycles found to visualize")
            return
            
        cycle_nodes = set()
        for cycle in cycles[:10]:
            cycle_nodes.update(cycle)
            
        if len(cycle_nodes) > max_nodes:
            cycle_nodes = set(list(cycle_nodes)[:max_nodes])
        
        G = nx.DiGraph()
        for node in cycle_nodes:
            for neighbor in self.graph[node]:
                if neighbor in cycle_nodes:
                    G.add_edge(node, neighbor)
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        nx.draw_networkx_edges(G, pos, edge_color='lightgray', arrows=True)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue')
        nx.draw_networkx_labels(G, pos)
        
        for i, cycle in enumerate(cycles[:10]):
            color = plt.cm.rainbow(i / min(10, len(cycles)))
            cycle_edges = list(zip(cycle, cycle[1:] + [cycle[0]]))
            nx.draw_networkx_edges(G, pos, edgelist=cycle_edges,
                                 edge_color=[color], arrows=True, width=2)
        
        plt.title(f"Gene Regulatory Network Cycles (Dataset {self.dataset_id})")
        if save_path:
            plt.savefig(save_path)
        plt.close()

class GRNNetworkClassifier:
    def __init__(self, embeddings: np.ndarray, threshold: float = 0.5):
        self.embeddings = embeddings
        self.threshold = threshold
        self.num_genes = embeddings.shape[0]
        self.graph = self._build_graph()
        
    def _build_graph(self) -> Dict[int, List[int]]:
        """Build adjacency list representation of the network."""
        graph = defaultdict(list)
        batch_size = min(1000, self.num_genes)
        
        for i in range(0, self.num_genes, batch_size):
            end_idx = min(i + batch_size, self.num_genes)
            batch_similarities = np.dot(self.embeddings[i:end_idx], self.embeddings.T)
            batch_similarities = (batch_similarities + 1) / 2
            
            for batch_idx, row in enumerate(batch_similarities):
                global_idx = i + batch_idx
                edges = np.where((row > self.threshold) & 
                               (np.arange(self.num_genes) != global_idx))[0]
                graph[global_idx].extend(edges.tolist())
        
        return graph

    def find_self_loops(self) -> List[int]:
        """Find all self-regulating genes."""
        self_loops = []
        similarity_matrix = np.dot(self.embeddings, self.embeddings.T)
        similarity_matrix = (similarity_matrix + 1) / 2
        
        for i in range(self.num_genes):
            if similarity_matrix[i,i] > self.threshold:
                self_loops.append(i)
        
        return self_loops

    def has_cycle(self) -> bool:
        """Check if network contains any cycles (excluding self-loops)."""
        visited = set()
        rec_stack = set()
        
        def dfs(node: int) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle, but check it's not just a self-loop
                    if neighbor != node:
                        return True
            
            rec_stack.remove(node)
            return False
        
        for node in range(self.num_genes):
            if node not in visited:
                if dfs(node):
                    return True
        
        return False

    def is_acyclic(self) -> bool:
        """Check if network is acyclic (excluding self-loops)."""
        return not self.has_cycle()

    def classify_network(self) -> Dict:
        """Classify network and provide detailed information."""
        self_loops = self.find_self_loops()
        has_cycles = self.has_cycle()
        
        classification = {
            'is_acyclic': not has_cycles,
            'has_cycles': has_cycles,
            'self_loops': self_loops,
            'num_self_loops': len(self_loops),
            'network_type': None
        }
        
        if has_cycles:
            classification['network_type'] = 'Cyclic Network'
        else:
            classification['network_type'] = 'Directed Acyclic Network (DAN)'
            
        return classification

import os
import pandas as pd
import networkx as nx
from typing import Dict, List, Set, Tuple
from collections import defaultdict

class SplitNetworkAnalyzer:
    def __init__(self, split: str, split_path: str):
        """Initialize analyzer for a specific split."""
        self.split = split
        self.split_path = split_path
        self.network = self._load_network()
        
    def _load_network(self) -> nx.DiGraph:
        """Load network from split's reference CSV file."""
        network_path = f'{self.split_path}/{self.split}/refNetwork.csv'
        
        if not os.path.exists(network_path):
            raise FileNotFoundError(f"Network file not found: {network_path}")
            
        # Read network file
        df = pd.read_csv(network_path)
        
        # Create directed graph
        G = nx.DiGraph()
        for _, row in df.iterrows():
            G.add_edge(row['Gene1'], row['Gene2'])
            
        return G
    
    def find_all_cycles(self) -> List[List[str]]:
        """Find all cycles in the network."""
        try:
            cycles = list(nx.simple_cycles(self.network))
            return cycles
        except Exception as e:
            print(f"Error finding cycles: {e}")
            return []
    
    def get_self_loops(self) -> Set[str]:
        """Get all genes with self-regulation."""
        return set(node for node, nbrs in self.network.adj.items() if node in nbrs)
    
    def analyze_network(self) -> Dict:
        """Perform comprehensive network analysis."""
        cycles = self.find_all_cycles()
        self_loops = self.get_self_loops()
        
        # Separate true cycles (length > 1) from self-loops
        true_cycles = [cycle for cycle in cycles if len(cycle) > 1]
        
        # Calculate in-degree and out-degree for each node
        in_degrees = dict(self.network.in_degree())
        out_degrees = dict(self.network.out_degree())
        
        stats = {
            'num_nodes': self.network.number_of_nodes(),
            'num_edges': self.network.number_of_edges(),
            'num_self_loops': len(self_loops),
            'num_true_cycles': len(true_cycles),
            'self_loops': list(self_loops),
            'true_cycles': true_cycles,
            'is_dag': nx.is_directed_acyclic_graph(self.network.copy()),
            'strongly_connected_components': list(nx.strongly_connected_components(self.network)),
            'cycle_lengths': [len(cycle) for cycle in true_cycles] if true_cycles else [],
            'max_in_degree': max(in_degrees.values()) if in_degrees else 0,
            'max_out_degree': max(out_degrees.values()) if out_degrees else 0,
            'avg_in_degree': sum(in_degrees.values()) / len(in_degrees) if in_degrees else 0,
            'avg_out_degree': sum(out_degrees.values()) / len(out_degrees) if out_degrees else 0
        }
        
        if true_cycles:
            stats.update({
                'max_cycle_length': max(stats['cycle_lengths']),
                'min_cycle_length': min(stats['cycle_lengths']),
                'avg_cycle_length': sum(stats['cycle_lengths']) / len(stats['cycle_lengths'])
            })
            
        return stats

def analyze_all_splits():
    """Analyze networks from all available splits."""
    for dataset in get_datasets():
        dataset_id = dataset['dataset_id']
        print(f"\nAnalyzing dataset DS{dataset_id}")
        splits_dir = f'./data/splits/DS{dataset_id}'
        if not os.path.exists(splits_dir):
            raise FileNotFoundError(f"Splits directory not found: {splits_dir}")
            
        splits = sorted([d for d in os.listdir(splits_dir) 
                        if os.path.isdir(os.path.join(splits_dir, d))])
        
        for split in splits:
            try:
                print(f"\nAnalyzing split: {split}")
                analyzer = SplitNetworkAnalyzer(split, splits_dir)
                stats = analyzer.analyze_network()
                
                print(f"\nNetwork Properties:")
                print(f"- Nodes: {stats['num_nodes']}")
                print(f"- Edges: {stats['num_edges']}")
                print(f"- Is DAG (Directed Acyclic Graph): {stats['is_dag']}")
                print(f"- Self-loops: {stats['num_self_loops']}")
                print(f"- True cycles: {stats['num_true_cycles']}")
                print(f"- Max in-degree: {stats['max_in_degree']}")
                print(f"- Max out-degree: {stats['max_out_degree']}")
                print(f"- Average in-degree: {stats['avg_in_degree']:.2f}")
                print(f"- Average out-degree: {stats['avg_out_degree']:.2f}")
                
                if stats['self_loops']:
                    print("\nSelf-regulating genes (first 10):")
                    for gene in sorted(stats['self_loops'])[:10]:
                        print(f"- {gene}")
                        
                if stats['true_cycles']:
                    print(f"\nTrue cycles statistics:")
                    print(f"- Maximum length: {stats['max_cycle_length']}")
                    print(f"- Minimum length: {stats['min_cycle_length']}")
                    print(f"- Average length: {stats['avg_cycle_length']:.2f}")
                    
                    print("\nExample true cycles (first 5):")
                    for i, cycle in enumerate(stats['true_cycles'][:5], 1):
                        print(f"{i}. {' -> '.join(cycle)} -> {cycle[0]}")
                        
                if stats['strongly_connected_components']:
                    large_components = [comp for comp in stats['strongly_connected_components'] 
                                    if len(comp) > 1]
                    if large_components:
                        print(f"\nStrongly connected components (size > 1):")
                        for i, comp in enumerate(large_components[:3], 1):
                            print(f"{i}. Size {len(comp)}: {', '.join(sorted(comp)[:5])}...")
                
                print("\n" + "="*50)
                
            except Exception as e:
                print(f"Error analyzing split {split}: {e}")
                continue


# if __name__ == "__main__":
#     datasets = get_datasets()
#     for dataset_info in datasets:  # Starting from index 7
#         try:
#             dataset_id = dataset_info['dataset_id']
#             print(f"Processing dataset {dataset_id}")
            
#             # Load embeddings
#             emb_path = f'./results/cl/DS{dataset_id}/pca_embeddings.npy'
#             if not os.path.exists(emb_path):
#                 print(f"Embeddings not found for dataset {dataset_id}")
#                 continue
                
#             embs = np.load(emb_path)
#             print(f"Loaded embeddings shape: {embs.shape}")
            
#             # Create detector with constraints
#             classifier = GRNNetworkClassifier(embs)
#             results = classifier.classify_network()
#             print(f"\nDataset {dataset_id} Analysis:")
#             print(f"Network Type: {results['network_type']}")
#             print(f"Is Acyclic: {results['is_acyclic']}")
#             print(f"Contains Cycles: {results['has_cycles']}")
#             print(f"Number of Self-loops: {results['num_self_loops']}")
            
#             if results['self_loops']:
#                 print("\nSelf-regulating genes:")
#                 print(results['self_loops'][:100])  # Show first 10

#             detector = GRNEmbbeddingCycleDetector(
#                 dataset_id=dataset_id,
#                 embeddings=embs,
#                 threshold=0.5,
#                 max_cycle_length=5,
#                 max_cycles=1000
#             )
            
#             # Get and print statistics
#             stats = detector.get_cycle_stats()
#             print(f"\nDataset {dataset_id} Analysis:")
#             print(f"Total cycles: {stats['total_cycles']}")
#             print(f"Average cycle length: {stats['avg_cycle_length']:.2f}")
#             print(f"Number of genes involved in cycles: {stats['genes_in_cycles']}")
#             print(f"Max cycle length: {stats['max_cycle_length']}")
#             print(f"Min cycle length: {stats['min_cycle_length']}")
            
#             # Visualize
#             detector.visualize_cycles(
#                 f'cycles_DS{dataset_id}.png',
#                 max_nodes=100
#             )
            
#             if stats['cycles']:
#                 print("\nExample cycles:")
#                 for i, cycle in enumerate(stats['cycles'][:3], 1):
#                     print(f"Cycle {i}: {' -> '.join(map(str, cycle + [cycle[0]]))}")
                    
#         except Exception as e:
#             print(f"Error processing dataset {dataset_id}: {e}")
#             continue


if __name__ == "__main__":
    analyze_all_splits()
