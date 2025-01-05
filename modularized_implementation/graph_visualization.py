# This file is not used in training the model. I created this file so that it would be easier for others
# to pick up the code and use it to visualize the expression graph. A step by step explanation of the 
# visualization process can be found in 'building_autograd_step_by_step/step_2_visualizing_graphs.ipynb'.

from graphviz import Digraph
from modularized_implementation.micro_grad import Value
from queue import Queue
from typing import Set, Tuple

def get_nodes_and_edges(root: Value) -> Tuple[Set[Value], Set[Tuple[Value, Value]]]:
    """Returns all the nodes and edges in the expression tree. Does not nodes for the operations.

    Args:
        root (Value): The root node (the final output object) of the expression tree.

    Returns:
        Tuple[Set[Value], Set[Tuple[Value, Value]]]: A tuple containing the set of nodes and set 
                                                     of edges in the expression tree.
    """
    nodes = set()
    edges = set()
    visited = set()
    queue = Queue()
    queue.put(root)
    while not queue.empty():
        node = queue.get()
        if node in visited:
            continue
        visited.add(node)
        nodes.add(node)
        if node.children:
            for child in node.children:
                edges.add((child, node))
                queue.put(child)
    return nodes, edges

def get_expression_graph(root: Value) -> Digraph:
    """Returns a graph that visualizes the expression created using the Value_4 objects.

    Args:
        root (Value): The root node (the final output object) of the expression tree.

    Returns:
        Digraph: DOT language graph that visualizes the expression created using the Value objects.
    """
    dot = Digraph(name="ExpressionGraph", 
                  comment="Constructs the expression graph using the Value objects.",
                  format="png",
                  graph_attr={"rankdir": "LR"})
    nodes, edges = get_nodes_and_edges(root)
    for node in nodes:
        unique_id: str = str(id(node))
        dot.node(name=unique_id, label=f"{node.label} | data {node.data:.4f} | grad {node.grad:.4f}", shape="record")
        if node.operation:
            dot.node(name=f"{unique_id}_{node.operation}", label=node.operation)
            dot.edge(tail_name=f"{unique_id}_{node.operation}", head_name=unique_id)
    for edge in edges:
        from_node, to_node = edge
        dot.edge(tail_name=str(id(from_node)), head_name=f"{str(id(to_node))}_{to_node.operation}")
    return dot