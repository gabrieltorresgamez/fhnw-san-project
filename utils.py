import os
import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from itertools import combinations
import numpy as np
import torch


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
    intermediaryNodes = intermediaryNodes[
        intermediaryNodes["sourceID"].str.contains(PAPERS)
    ]
    officerNodes = officerNodes[officerNodes["sourceID"].str.contains(PAPERS)]
    nodes_others = nodes_others[nodes_others["sourceID"].str.contains(PAPERS)]

    # alternatively, get all nodeIDs from all filtered nodes and remove relationships with nodeIDs that are not in this list
    allNodeIDs = pd.concat(
        [addressNodes, entityNodes, intermediaryNodes, officerNodes, nodes_others]
    ).index
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
        [
            (*relationships.index[i], value)
            for i, value in enumerate(relationships.to_dict(orient="records"))
        ]
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
                G.subgraph(
                    [
                        node
                        for node in component
                        if G.nodes[node]["node_type"] == "Address"
                    ]
                ),
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


def is_officer_from_country_name(G, node_id, country_name):
    """Check if a node is an officer from a specific country."""
    node = G.nodes[node_id]
    return node["node_type"] == "Officer" and country_name in node["countries"]


def filter_nodes(G: nx.Graph, query: str):
    """Filters nodes by SQL-like query.
    Args:
        - G (nx.Graph): graph
        - query (str): SQL query
    Returns:
        numpy array with node ids
    """
    return pd.DataFrame.from_dict(G.nodes, orient="index").query(query).index.to_numpy()


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

    # only store node ids which are contracted onto target
    target_attr = G.nodes[target]
    target_attr["contraction"] = list(target_attr.get("contraction", {}).keys())
    nx.set_node_attributes(G, {target: target_attr})

    return G, target


def merge_duplicate_nodes(graph, exclude_attributes=["label"]):
    """
    Removes duplicate edges from a graph. An edge is considered a duplicate if it has
    the same source, target, and 'link' attribute as another edge. Only the first occurrence
    of a duplicate edge is retained.

    Parameters:
        graph (networkx.Graph): The graph from which duplicate edges are to be removed.
                                 This graph is not modified; a new graph is returned.

    Returns:
        networkx.Graph: A new graph instance that is a copy of the original but with all
                        duplicate edges removed. This ensures that each edge is unique
                        based on the combination of source, target, and 'link' attribute.

    Note:
        This function handles graphs with multiple edges between the same nodes by comparing
        the 'link' attribute. It assumes that this attribute exists for all edges. If it does
        not, the function may behave unexpectedly.
    """

    # Convert graph nodes to DataFrame and copy the graph.
    updated_graph = graph.copy()
    node_attributes = pd.DataFrame.from_dict(
        dict(updated_graph.nodes(data=True)), orient="index"
    )

    # Normalize attribute values for consistent comparison.
    replace_dict = {
        "_": " ",
        "-": " ",
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "ß": "ss",
    }
    node_attributes = node_attributes.map(
        lambda x: (
            "".join(replace_dict.get(c, c) for c in x.lower())
            if isinstance(x, str)
            else x
        )
    )

    # Group nodes, excluding specified attributes for comparison.
    relevant_attributes = node_attributes.columns.difference(exclude_attributes)
    grouped_nodes = node_attributes.groupby(list(relevant_attributes), dropna=False)

    # Merge nodes with identical attributes.
    for _, nodes_in_group in tqdm(
        grouped_nodes, desc="Merging duplicate nodes", total=len(grouped_nodes)
    ):
        if len(nodes_in_group) > 1:
            primary_node = nodes_in_group.index[0]
            for node_id in nodes_in_group.index[1:]:
                nx.contracted_nodes(
                    updated_graph, primary_node, node_id, self_loops=False, copy=False
                )

    return updated_graph


def remove_duplicate_edges(graph):
    """
    Removes duplicate edges from a graph based on a specific attribute within each edge.

    Parameters:
        graph (networkx.Graph): The graph from which duplicate edges will be removed.

    Returns:
        networkx.Graph: A new graph with duplicate edges removed based on the 'link' attribute.

    Notes:
        This function identifies duplicates by creating a unique signature for each edge based
        on the 'source', 'target', and 'link' attribute. It collects all unique edges and removes
        any additional edges that have the same signature.
    """
    updated_graph = graph.copy()
    existing_edges = set()
    redundant_edges = []

    # Iterate over all edges and identify duplicates.
    for source, target, key, data in tqdm(
        updated_graph.edges(data=True, keys=True), desc="Removing duplicate edges"
    ):
        edge_signature = (source, target, data["link"])

        # Check and mark duplicate edges.
        if edge_signature in existing_edges:
            redundant_edges.append((source, target, key))
        else:
            existing_edges.add(edge_signature)

    # Remove identified duplicate edges.
    for source, target, key in redundant_edges:
        updated_graph.remove_edge(source, target, key)

    return updated_graph


def get_swiss_officer_entities_subgraph(G):
    swiss_officers = filter_nodes(
        G, query="countries == 'Switzerland' and node_type == 'Officer'"
    )
    all_entities = filter_nodes(G, query="node_type == 'Entity'")

    swiss_officers_entities_subgraph_ = G.subgraph(
        set(swiss_officers) | set(all_entities)
    )
    filtered_edges_u_v_k = [
        (u, v, k)
        for u, v, k in swiss_officers_entities_subgraph_.edges(keys=True)
        if u in swiss_officers and v in all_entities
    ]
    swiss_officers_entities_subgraph = G.edge_subgraph(filtered_edges_u_v_k)

    return swiss_officers_entities_subgraph, swiss_officers, all_entities


def plot_ego_with_labels(G, node, color_map, ego_radius=1, plot_type_circular=True):
    ego = nx.ego_graph(G, node, radius=ego_radius, undirected=True)
    pos = (
        nx.circular_layout(ego) if plot_type_circular else nx.spring_layout(ego, k=0.5)
    )
    colors = [color_map[G.nodes[n]["node_type"]] for n in ego.nodes]
    labels = {
        n: (
            G.nodes[n]["name"]
            if G.nodes[n]["node_type"] != "Address"
            else G.nodes[n]["address"].split(",")[0]
        )
        for n in ego.nodes
    }

    edge_labels = {}
    for u, v, d in ego.edges(data=True):
        key = (u, v)
        if key not in edge_labels:
            edge_labels[key] = []
        edge_labels[key].append(d["link"])

    combined_edge_labels = {k: "\n".join(v) for k, v in edge_labels.items()}

    nx.draw(
        ego,
        pos,
        with_labels=True,
        node_color=colors,
        labels=labels,
        font_size=plt.rcParams["font.size"],
        edge_color="#D3D3D3",
    )
    nx.draw_networkx_edge_labels(
        ego,
        pos,
        edge_labels=combined_edge_labels,
        font_color="red",
        font_size=plt.rcParams["font.size"] - 1,
    )

    plt.title(f"Ego graph of {labels[node]}: {node}")
    plt.suptitle(
        "Entity = red, Officer = blue, Intermediary = green, Address = yellow, Other = gray",
        y=0.01,
        fontsize=plt.rcParams["font.size"],
        color="gray",
    )
    plt.show()


def global_view(G: nx.Graph, by: str, self_loops=False):
    """Create a global view of the network grouping by a certain attribute.

    Params:
        - G (nx.Graph): networkx Grap
        - by (str): node attribute to groupe by
        - self_loops (bool): self loops allowed or not

    Returns:
        modified graph, mapper which maps the group to its node
    """

    G = nx.MultiGraph(G.copy())

    # iterate over the unique expressions of the 'by' attribute
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
        # get weight of edge if exists otherwise use default of 1
        edge_weight = data.get("weight", 1)

        # If the edge already exists in the new graph, increment its weight
        if G_new.has_edge(u, v):
            G_new[u][v]["weight"] += edge_weight
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


def graph_from_attr_name(G: nx.Graph, attribute_name: str) -> nx.Graph:
    """
    Create a new graph based on a given node attribute from an existing graph.

    This function generates a new graph where edges are created between nodes
    that share the same value for a specified attribute in the original graph.
    Nodes without any connections in the new graph are still included.

    Parameters:
    -----------
    G : nx.Graph
        The original graph from which the new graph is created.
    attribute_name : str
        The name of the attribute based on which the new graph's edges are formed.

    Returns:
    --------
    nx.Graph
        A new graph where nodes sharing the same attribute value are connected.
    """

    # Extract the specified attribute from the original graph's nodes into a DataFrame
    attribute = pd.DataFrame.from_dict(
        nx.get_node_attributes(G, attribute_name), orient="index", columns=["attr"]
    )

    # Initialize an empty list to hold the edge pairs
    edgelist = []

    # Group nodes by the specified attribute value and create edges between all nodes in each group
    for _, df in attribute.groupby("attr"):
        combinations_indices = list(combinations(df.index, 2))
        edgelist.extend(combinations_indices)

    # Create a new graph from the generated edge list
    G_new = nx.from_edgelist(edgelist)

    # Add nodes that were in the original graph but have no edges in the new graph
    G_new.add_nodes_from(np.setdiff1d(list(G.nodes()), list(G_new.nodes())))

    # Copy all node attributes from the original graph to the new graph
    nx.set_node_attributes(G_new, dict(G.nodes(data=True)))

    return G_new


def pearson_correlation(x: torch.Tensor, y: torch.Tensor):
    """
    Compute the Pearson correlation coefficient between two tensors x and y.

    Parameters:
    x (torch.Tensor): The first input tensor.
    y (torch.Tensor): The second input tensor.

    Returns:
    torch.Tensor: The Pearson correlation coefficient.
    """

    mean_x = torch.nanmean(x, dtype=torch.float32)
    mean_y = torch.nanmean(y, dtype=torch.float32)
    numerator = torch.nansum((x - mean_x) * (y - mean_y), dtype=torch.float32)
    denominator = torch.sqrt(
        torch.nansum((x - mean_x) ** 2, dtype=torch.float32)
        * torch.nansum((y - mean_y) ** 2, dtype=torch.float32)
    )
    return numerator / denominator


def dyadic_hypothesis_test(
    G1: nx.Graph,
    G2: nx.Graph,
    metric: callable = lambda x, y: pearson_correlation(x, y),
    n: int = 10000,
    dtype=torch.float16,
    device="cuda",
    self_loops: bool = False,
):
    """
    Perform a dyadic hypothesis test between two graphs G1 and G2 using a specified metric.

    Parameters:
    G1 (nx.Graph): The first input graph.
    G2 (nx.Graph): The second input graph, which must have the same nodes as G1.
    metric (callable): A function to compute the similarity metric between the graphs. Default is Pearson correlation coefficient.
    n (int): The number of permutations to perform for the test. Default is 10,000.
    dtype (type): The data type to use for adjacency matrices. Default is torch.float16.
    device (str): The device to use for tensor computations. Default is "cuda".
    self_loops (bool): Whether to include self-loops (diagonal elements) in the adjacency matrices. Default is False.

    Returns:
    tuple: original metric, dataframe containing the metrics of the permuted graphs
    """

    # Ensure G1 and G2 have the same number of nodes
    assert (
        G1.number_of_nodes() == G2.number_of_nodes()
    ), "G1 and G2 must have the same number of nodes"

    # Ensure G1 and G2 have the same nodes
    assert (
        np.array(sorted(G1.nodes())) == np.array(sorted(G2.nodes()))
    ).all(), "G1 and G2 must have the same nodes"

    # Get node order
    nodes = list(G1.nodes())

    # Create adjacency matrix for G1 and send to device
    G1_vector = torch.tensor(
        nx.adjacency_matrix(G1, nodes).toarray(), dtype=dtype, device=device
    )

    # ignoring self loops activated -> set diagonal to nan
    if not self_loops:
        G1_vector = G1_vector.fill_diagonal_(float("nan"))

    # make to vector
    G1_vector = G1_vector.flatten()

    # Calculate adjacency matrix of G2 and convert to torch tensor
    G2_adj = torch.tensor(
        nx.adjacency_matrix(G2, nodes).toarray(), dtype=dtype, device=device
    )

    def apply_metric(G2_adj: torch.Tensor):
        """
        Apply the specified metric to the vectorized form of G1 and the flattened G2 adjacency matrix.

        Parameters:
        G2_adj (torch.Tensor): The adjacency matrix of G2.

        Returns:
        torch.Tensor: The computed metric value.
        """

        # ignoring self loops activated -> set diagonal to nan
        if not self_loops:
            G2_vector = G2_adj.fill_diagonal_(float("nan"))

        # make to vector
        G2_vector = G2_vector.flatten()

        return metric(G1_vector, G2_vector).cpu().item()

    # Calculate metric of original graph
    metric_original = apply_metric(G2_adj)

    def QAP(adj_matrix: torch.Tensor):
        """
        Perform a Quadratic Assignment Procedure (QAP) by randomly permuting the rows and columns of the adjacency matrix.

        Parameters:
        adj_matrix (torch.Tensor): The adjacency matrix to permute.

        Returns:
        torch.Tensor: The permuted adjacency matrix.
        """
        # Define random row and column permutations for G2_adj
        row_permutation = torch.randperm(adj_matrix.shape[0], device=device)
        col_permutation = torch.randperm(adj_matrix.shape[1], device=device)

        # Perform permutation and return
        return adj_matrix[row_permutation, :][:, col_permutation]

    # Permute G2_adj and calculate metric n times
    metric_runs = []
    for _ in tqdm(range(n)):
        # Apply QAP and calculate metric to permuted adjacency matrix
        metric_runs.append(apply_metric(QAP(G2_adj)))

    return metric_original, pd.DataFrame({"metric_permuted": metric_runs})
