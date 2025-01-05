from typing import Any, List, Set

def topological_sort(root: Any) -> List[Any]:
    visited: Set[Any] = set()
    topo_order: List[Any] = []
    def dfs(node: Any):
        if node in visited:
            return
        visited.add(node)
        if node.children:
            for child in node.children:
                dfs(child)
        topo_order.append(node)
    dfs(root)
    topo_order.reverse()
    return topo_order