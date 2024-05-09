import os
import re
import pandas as pd
import networkx as nx
from multiprocessing.dummy import Pool as ThreadPool


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


def filter_nodes_by_attr(node_attrs, attr_value):
    """Filter nodes based on a specific attribute value.

    Args:
        node_attrs (dict): A dictionary containing node attributes.
        attr_value (str): The attribute value to match.

    Returns:
        set: A set of nodes that match the specified attribute value.
    """
    return set(node for node, value in node_attrs.items() if value == attr_value)


def global_view(G: nx.Graph, attr: list, self_loops=False):
    """Contracts nodes which follow the conditions of attributes to a single node using multithreading.

    Args:
        - G (nx.Graph): The input graph.
        - attr (list of tuples): Example [('node_type', 'Officer'), ('country', 'CHE')] 
            -> All nodes of type 'Officer' with 'CHE' as the country will be contracted into one (can be extended further).
        - self_loops (bool): Whether or not the contracted nodes should have self-loops if they had connections originally.

    Returns:
        tuple: The modified graph and the contracted node.
    """

    # Extract all node attributes once for filtering
    all_node_attrs = {attr_name: nx.get_node_attributes(G, attr_name) for attr_name, _ in attr}

    # Helper function to process attributes
    def filter_fn(args):
        attr_name, attr_value = args
        return filter_nodes_by_attr(all_node_attrs[attr_name], attr_value)

    # Use a ThreadPool instead of a Multiprocessing Pool to handle parallel filtering
    with ThreadPool() as pool:
        results = pool.map(filter_fn, attr)

    # Find the intersection of all filtered sets
    nodes = set.intersection(*results) if results else set()

    # If no nodes match the criteria, return the original graph and None
    if not nodes:
        return G, None

    # Convert to a list for easier iteration
    nodes = list(nodes)

    # Make a copy of the graph only if we're actually contracting nodes
    G = G.copy()
    target = nodes[0]

    # Contract nodes to the target node
    for node in nodes[1:]:
        nx.contracted_nodes(G, target, node, self_loops=self_loops, copy=False)

    return G, target
