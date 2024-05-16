import os
import re
import pandas as pd
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
import numpy as np



def __remove_special_characters(text):
    # Stellen Sie sicher, dass der Text ein String ist
    return re.sub(r"[^\w\s;,.]", "", text) if isinstance(text, str) else text


def get_graph(GRAPH_PATH, PAPERS="Pandora"):
    """
    Load the graph from the GRAPH_PATH if it exists, otherwise create the graph from the data in the data folder.
    """

    if os.path.exists(GRAPH_PATH):
        return nx.read_gexf(GRAPH_PATH)

    # read all the data
    addressNodes = (
        pd.read_csv("./data/nodes-addresses.csv", low_memory=False, index_col=0)
        .astype(str)
        .map(__remove_special_characters)
    )
    addressNodes["node_type"] = "Address"

    entityNodes = (
        pd.read_csv("./data/nodes-entities.csv", low_memory=False, index_col=0)
        .astype(str)
        .map(__remove_special_characters)
    )
    entityNodes["node_type"] = "Entity"

    intermediaryNodes = (
        pd.read_csv("./data/nodes-intermediaries.csv", low_memory=False, index_col=0)
        .astype(str)
        .map(__remove_special_characters)
    )
    intermediaryNodes["node_type"] = "Intermediary"

    officerNodes = (
        pd.read_csv("./data/nodes-officers.csv", low_memory=False, index_col=0)
        .astype(str)
        .map(__remove_special_characters)
    )
    officerNodes["node_type"] = "Officer"

    nodes_others = (
        pd.read_csv("./data/nodes-others.csv", low_memory=False, index_col=0)
        .astype(str)
        .map(__remove_special_characters)
    )
    nodes_others["node_type"] = "Other"

    relationships = (
        pd.read_csv("./data/relationships.csv", low_memory=False)
        .set_index(["node_id_start", "node_id_end"])
        .astype(str)
        .map(__remove_special_characters)
    )

    # filter all nodes and relationships that are not from the Pandora Papers
    addressNodes = addressNodes[addressNodes["sourceID"].str.contains(PAPERS)]
    entityNodes = entityNodes[entityNodes["sourceID"].str.contains(PAPERS)]
    intermediaryNodes = intermediaryNodes[intermediaryNodes["sourceID"].str.contains(PAPERS)]
    officerNodes = officerNodes[officerNodes["sourceID"].str.contains(PAPERS)]
    nodes_others = nodes_others[nodes_others["sourceID"].str.contains(PAPERS)]

    # alternatively, get all nodeIDs from all filtered nodes and remove relationships with nodeIDs that are not in this list
    allNodeIDs = pd.concat([addressNodes, entityNodes, intermediaryNodes, officerNodes, nodes_others]).index
    relationships = relationships[
        relationships.index.get_level_values(0).isin(allNodeIDs)
        & relationships.index.get_level_values(1).isin(allNodeIDs)
    ]

    # create the graph
    G = nx.MultiDiGraph()
    G.add_nodes_from(list(addressNodes.to_dict("index").items()), bipartite=0)
    G.add_nodes_from(list(entityNodes.to_dict("index").items()), bipartite=1)
    G.add_nodes_from(list(intermediaryNodes.to_dict("index").items()), bipartite=2)
    G.add_nodes_from(list(officerNodes.to_dict("index").items()), bipartite=3)
    G.add_nodes_from(list(nodes_others.to_dict("index").items()), bipartite=4)
    G.add_edges_from(
        [(*relationships.index[i], value) for i, value in enumerate(relationships.to_dict(orient="records"))]
    )

    # remove all the dataframes
    del addressNodes
    del entityNodes
    del intermediaryNodes
    del officerNodes
    del nodes_others
    del relationships

    # save the graph
    nx.write_gexf(G, GRAPH_PATH)

    return G


def filter_graph_by_country_name(G, country_name, verbose=True):
    """
    Return a Graph, which contains only the connected components that contain min. 1 node with the country_name.
    """

    if verbose:
        print("Info pre filtering:")
        print("Number of edges: ", G.number_of_edges())
        print("Number of nodes: ", G.number_of_nodes())
        print(
            "Number of weakly connected components: ",
            nx.number_weakly_connected_components(G),
        )

        print(f'\nFiltering the graph after the country "{country_name}"\n')

    # reduce graph to connected components which contain a swiss address
    swiss_components = []
    for component in nx.connected_components(G.to_undirected()):
        if country_name in list(
            nx.get_node_attributes(
                G.subgraph([node for node in component if G.nodes[node]["node_type"] == "Address"]),
                "countries",
            ).values()
        ):
            swiss_components.append(component)

    reduced_G = G.subgraph(set.union(*swiss_components))

    if verbose:
        print("Info post filtering:")
        print("Number of edges: ", reduced_G.number_of_edges())
        print("Number of nodes: ", reduced_G.number_of_nodes())
        print(
            "Number of weakly connected components: ",
            nx.number_weakly_connected_components(reduced_G),
        )

    return reduced_G


def filter_nodes(G: nx.Graph, query: str):
    """Filters nodes by SQL-like query.
    Args:
        - G (nx.Graph): graph
        - query (str): SQL query
    Returns:
        numpy array with node ids
    """
    return pd.DataFrame.from_dict(G.nodes, orient='index').query(query).index.to_numpy()

  
def is_officer_from_country_name(G, node_id, country_name):
    """Check if a node is an officer from a specific country."""
    node = G.nodes[node_id]
    return node["node_type"] == "Officer" and country_name in node["countries"]


def filter_nodes_by_attributes(G: nx.Graph, attr: list):
    """Filters nodes by attributes

    Args:
        - G (nx.Graph): The input graph.
        - attr (list of tuples): Example [('node_type', 'Officer'), ('country', 'CHE')]
            -> All nodes of type 'Officer' with 'CHE' as the country will be contracted into one (can be extended further).
            -> Attribution value is also allowed to be a list to allow matching multiple values
            
    Returns:
        numpy array with node ids
    """

    # Extract all node attributes once for filtering
    all_node_attrs = {attr_name: nx.get_node_attributes(G, attr_name) for attr_name, _ in attr}

    # Helper function to process attributes
    def filter_fn(args):
        attr_name, attr_value = args

        # if attr_value is a list (multiple values allowed)
        if isinstance(attr_value, list):
            return {node for node, value in all_node_attrs[attr_name].items() if value in attr_value}

        return {node for node, value in all_node_attrs[attr_name].items() if value == attr_value}


def contract_nodes(G: nx.Graph, query: str, self_loops=False):
    """Contracts nodes which follow the conditions of attributes to a single node using pandas filtering.

    Args:
        - G (nx.Graph): The input graph.
        - query (str): SQL-like query string to filter nodes.
        - self_loops (bool): Whether or not the contracted nodes should have self-loops if they had connections originally.

    Returns:
        tuple: The modified graph and the contracted node.
    """

    # Get all nodes that match the query
    nodes = filter_nodes(G, query)

    # If no nodes match the criteria, return the original graph and None
    if not len(nodes):
        return G, None

    # Create a copy of the graph to perform modifications
    G = G.copy()
    target = nodes[0]  # The first node serves as the target node for the contraction

    # contract nodes to target
    for node in nodes[1:]:
        nx.contracted_nodes(G, target, node, self_loops, copy=False)

    #only store node ids which are contracted onto target
    target_attr = G.nodes[target]
    target_attr["contraction"] = list(target_attr.get("contraction", {}).keys())
    nx.set_node_attributes(G, {target:target_attr})
        
    return G, target


def global_view(G: nx.Graph, by:str, self_loops=False):
    """Create a global view of the network grouping by a certain attribute.
    
    Params:
        - G (nx.Graph): networkx Grap
        - by (str): node attribute to groupe by
        - self_loops (bool): self loops allowed or not

    Returns:
        modified graph, mapper which maps the group to its node
    """

    G = nx.MultiGraph(G.copy())

    #iterate over the unique expressions of the 'by' attribute
    mapper = {}
    for by_attr in np.unique(list(nx.get_node_attributes(G, by).values())):
        G, target_node = contract_nodes(G, f"{by} == '{by_attr}'", self_loops)
        mapper[by_attr] = target_node

    return G, mapper


def multigraph_to_graph(G: nx.MultiGraph) -> nx.Graph:
    """
    Convert a MultiGraph to a Graph, merging multiple edges between the same nodes into a single edge.
    The weight of the edge in the resulting graph represents the number of multiple edges in the original MultiGraph.

    Parameters:
    G (nx.MultiGraph): The input MultiGraph to be converted.

    Returns:
    nx.Graph: A Graph where each pair of nodes is connected by a single edge with a weight representing
              the count of original edges between those nodes in the MultiGraph.
    """
    # Create a new simple graph to hold the converted structure
    G_new = nx.Graph()
    
    # Iterate over each edge in the MultiGraph
    for u, v, data in G.edges(data=True):
        #get weight of edge if exists otherwise use default of 1
        edge_weight = data.get("weight", 1)

        # If the edge already exists in the new graph, increment its weight
        if G_new.has_edge(u, v):
            G_new[u][v]['weight'] += edge_weight
            continue
        
        # Add a new edge with initial weight if it does not exist
        G_new.add_edge(u, v, weight=edge_weight)

    # Add nodes not added yet
    G_new.add_nodes_from(np.setdiff1d(G.nodes, G_new.nodes))

    # Copy all node attributes from the original MultiGraph to the new Graph
    nx.set_node_attributes(G_new, dict(G.nodes(data=True)))
    
    return G_new


def permute_graph_QAP(G: nx.Graph, self_loops=False) -> nx.Graph:
    """
    Permute the nodes of the given graph G by random row and column swaps of its adjacency matrix,
    and return a new graph with the same nodes but with the permuted structure.

    Parameters:
    G (nx.Graph): The input graph to be permuted.
    self_loops (bool): True -> self loops allowed, False -> weights of self loops are set to zero

    Returns:
    nx.Graph: A new graph with the permuted adjacency matrix, preserving the original node attributes.
    """
    # Get the list of nodes in the graph
    nodelist = list(G.nodes())

    # Get the adjacency matrix from the graph
    adj_matrix = nx.adjacency_matrix(G, nodelist).todense()

    # Define random row and column permutations
    row_permutation = np.random.permutation(np.arange(adj_matrix.shape[0]))
    col_permutation = np.random.permutation(np.arange(adj_matrix.shape[1]))

    # Apply the permutations to the adjacency matrix
    adj_matrix = adj_matrix[row_permutation, :][:, col_permutation]

    # if self loops disabled -> set diag to zero
    if not self_loops:
        np.fill_diagonal(adj_matrix, 0)

    # Create a new graph from the permuted adjacency matrix
    new_G = nx.from_numpy_array(adj_matrix)

    # Relabel the nodes so that the original node IDs are used
    nx.relabel_nodes(new_G, lambda i: nodelist[i], copy=False)

    # Preserve the node attributes from the original graph
    nx.set_node_attributes(new_G, dict(G.nodes(data=True)))

    return new_G