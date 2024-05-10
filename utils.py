import os
import re
import pandas as pd
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random



def __remove_special_characters(text):
    # Stellen Sie sicher, dass der Text ein String ist
    if isinstance(text, str):
        # Entfernt alles außer Buchstaben, Ziffern, Leerzeichen und grundlegenden Satzzeichen
        return re.sub(r"[^\w\s,.]", "", text)
    return text


def get_graph(GRAPH_PATH, PAPERS="Pandora"):
    """
    Load the graph from the GRAPH_PATH if it exists, otherwise create the graph from the data in the data folder.
    """

    if not os.path.exists(GRAPH_PATH):
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
            pd.read_csv(
                "./data/nodes-intermediaries.csv", low_memory=False, index_col=0
            )
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
        G.add_nodes_from(
            [(key, value) for key, value in addressNodes.to_dict("index").items()],
            bipartite=0,
        )
        G.add_nodes_from(
            [(key, value) for key, value in entityNodes.to_dict("index").items()],
            bipartite=1,
        )
        G.add_nodes_from(
            [(key, value) for key, value in intermediaryNodes.to_dict("index").items()],
            bipartite=2,
        )
        G.add_nodes_from(
            [(key, value) for key, value in officerNodes.to_dict("index").items()],
            bipartite=3,
        )
        G.add_nodes_from(
            [(key, value) for key, value in nodes_others.to_dict("index").items()],
            bipartite=4,
        )
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

    return nx.read_gexf(GRAPH_PATH)


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


def filter_nodes(G: nx.Graph, query: str):
    """Filters nodes by SQL-like query.
    Args:
        - G (nx.Graph): graph
        - query (str): SQL query

    Returns:
        numpy array with node ids
    """
    return pd.DataFrame.from_dict(G.nodes, orient='index').query(query).index.to_numpy()


def global_view(G: nx.Graph, query: str, self_loops=False):
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

    # Get the set of edges related to the contracted nodes
    contracted_edges = []
    for node in nodes:
        for neighbor in G.neighbors(node):
            if neighbor not in nodes or (self_loops and neighbor == target):
                contracted_edges.append((target, neighbor))
    
    # Remove all contracted nodes except the target
    G.remove_nodes_from(set(nodes) - {target})

    # Add the consolidated edges to the target node
    G.add_edges_from(contracted_edges)

    return G, target


def compute_degree_for_country_explicit(args) -> tuple:
    """Hilfsfunktion, um den Degree für ein bestimmtes Land zu berechnen.

    Args:
        - args (tuple): Ein Tupel, das die Graph-Daten (als dict) und den Ländercode enthält.

    Returns:
        tuple: Ein Tupel mit dem Ländercode und dem berechneten Degree.
    """
    G_data, cc = args

    # Den Graphen aus dem übergebenen Dictionary wiederherstellen
    G = nx.node_link_graph(G_data)

    # Erstelle eine Abfrage für den aktuellen Ländercode
    query = f"country_codes == '{cc}'"

    # Rufe global_view_pandas mit der gegebenen Abfrage auf
    G_, target_node = global_view(G, query)

    # Berechne den Degree des Zielknotens
    degree = nx.degree(G_, target_node) / (G_.number_of_nodes() - 1) if G_.number_of_nodes() > 1 else 0

    # Rückgabe des Ländercodes und des berechneten Werts
    return (cc, degree)


def get_degree_by_country_code_parallel(G: nx.Graph):
    """Berechnet die Degree-Werte für jeden Ländercode parallel (mit Prozessen).

    Args:
        - G (nx.Graph): Der Eingabegraph.

    Returns:
        dict: Ein Wörterbuch mit den Ländercodes und den berechneten Degree-Werten.
    """
    # Alle eindeutigen Ländercodes im Graphen finden
    country_codes = np.unique(list(nx.get_node_attributes(G, "country_codes").values()))

    # Konvertiere den Graphen in ein serialisierbares Format
    G_data = nx.node_link_data(G)

    # Erstelle eine Liste der Argumente (G_data und Ländercode) für jeden Prozess
    args_list = [(G_data, cc) for cc in country_codes]

    # Verwende einen ProcessPoolExecutor für die parallele Verarbeitung
    with ProcessPoolExecutor() as executor:
        # Führe `compute_degree_for_country_explicit` für jeden Ländercode parallel aus
        results = executor.map(compute_degree_for_country_explicit, args_list)

    # Sammle die Ergebnisse in einem Wörterbuch
    return dict(results)


def create_random_edges_from(G: nx.Graph):
    """Creates a random graph by maintaining the same nodes as the input graph 
    but with randomly selected edges.

    Args:
        G (nx.Graph): The original input graph.

    Returns:
        nx.Graph: A new graph with the same nodes but random edges.
    """
    # Create a new graph with the same nodes and attributes
    new_G = nx.Graph()
    new_G.add_nodes_from([(key, value) for key, value in G.nodes.items()])

    # Determine the number of edges in the original graph
    num_edges = G.number_of_edges()

    # Convert the nodes to a list to ensure compatibility with random.sample
    node_list = list(new_G.nodes)

    # Set to track unique edges
    existing_edges = set()

    # Add random edges until we reach the desired count
    while len(existing_edges) < num_edges:
        # Select two random nodes to form an edge
        u, v = random.sample(node_list, 2)
        # Ensure the edge isn't already added (in either direction)
        if (u, v) not in existing_edges and (v, u) not in existing_edges:
            existing_edges.add((u, v))

    # Add the randomly selected edges to the new graph
    new_G.add_edges_from(existing_edges)

    return new_G