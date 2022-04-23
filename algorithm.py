from itertools import combinations
from random import random, choice
import networkx as nx
from time import perf_counter

def t(start):
    return round(perf_counter() - start, 7)

def query(u, v):
    return u < v

def random_graph(n, p):
    """
    Create Erdos Renyi random graph G(n, p)
    with planted Hamiltonian path
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Include Hamiltonian path
    for i in range(n):
        G.add_edge(i, (i+1) % n)

    edges = combinations(range(n), 2)
    for e in edges:
        if random() < p:
            G.add_edge(*e)
    return G

def find_min(graph: nx.Graph):
    """
    Find the first smallest element
    """
    min_elem = None
    s = set()
    edge = choice(list(graph.edges))
    u, v = edge
    if query(u, v):
        min_elem = u
        s.add(v)
    else:
        min_elem = v
        s.add(u)
    
    # only neighbors can be the next min
    current_neighbors = graph[min_elem]
    while True:
        for u in current_neighbors:
            still_adding = False
            if u not in s:
                # u may potentially be the smallest element
                still_adding = True
                if query(u, min_elem):
                    # we established u < min_elem so we break here
                    # to stop looking at min_elem's neighbors,
                    # and we move onto the neighbors of u
                    s.add(min_elem)
                    min_elem = u
                    current_neighbors = graph[min_elem]
                    break
                else:
                    # u is not the smallest element
                    s.add(u)
        if not still_adding:
            # if we have not added a new element to s,
            # it means we found the smallest element
            break
    return min_elem

def create_level(levels, elim, i):
    """
    This one is taking the most amount of time
    (probably because of the .copy() function)
    """
    while i >= 0:
        levels[i] = levels[i+1].copy()
        if len(levels[i]) == 0:
            i -= 1
            continue
        for v in levels[i+1]:
            elim[v] = None
            # in paper it's not last level but level i+c but I removed c
            # for ease of implementation, I can always add it later
            for u in levels[-1]: # topmost level containing all the vertices
                if query(u, v):
                    if v in levels[i]:
                        # should not have this if I don't think but okay
                        levels[i].remove(v)
                    elim[v] = u
        
        i -= 1
    return levels

def lowest_level_containing(node, levels):
    for i, level in enumerate(levels):
        if node in level:
            return i

def increment(graph: nx.Graph, last_min, levels, elim):
    """
    Incremental updates to levels
    """
    # the previous min is no longer a part of the levels
    for level in levels:
        if last_min in level:
            level.remove(last_min)

    for v in graph.nodes:
        # For each node that was blocked by last_min, do smt
        if elim[v] != last_min:
            continue
        elim[v] = None
        i = lowest_level_containing(v, levels)
        while i > 1:
            # again, in paper it's not last level but level i+c 
            # but I removed c for ease of implementation
            for u in levels[-1]:
                if query(u, v):
                    elim[v] = u
                    break
            i = i - 1
            levels[i-1].append(v)
        
def find(graph: nx.Graph, last_min, levels, current_sorting, q):
    """
    Find the next min element
    """
    # Make the candidate set - neighbors of last_min!!!!!!!
    neighbors = list(graph[last_min])
    s = set()
    for v in neighbors:
        if v not in current_sorting and v in levels[0]:
            # Only if not already sorted, and it's in the last level,
            # AND it's a neighbor to last min, it's a candidate
            # the next min element is guaranteed to be here
            # because of the included Hamiltonian path
            s.add(v)
    
    if len(s) == 1:
        for elem in s:
            break
        return elem

    # This loop actually never happens lmao
    for i in range(q):
        for v in s: 
            for u in levels[i]:
                if (u, v) in graph.edges:
                    if query(u, v):
                        s.remove(v)
                if len(s) == 1:
                    for elem in s:
                        break
                    return elem

def stochastic_sort(graph: nx.Graph, p, q):
    """
    I removed the "c" parameter, I just have 
    q levels that I will maintain and I have the q+1
    level to have all the nodes, and likewise level q+2
    I will increase this if necessary
    """
    # Set up
    levels = []
    n = graph.number_of_nodes()
    start = perf_counter()
    for i in range(q+1): # this many levels
        if i < q-1:
            levels.append([]) # the idea is to construct the lower levels from the top ones
        else:
            levels.append(list(graph.nodes)) # the topmost levels have all the nodes

    print(f"\tMaking \"levels\": {t(start)}s")
    elim = [None] * n
    start = perf_counter()
    levels = create_level(levels, elim, q-1) # create bottom levels
    print(f"\tCreated all levels: {t(start)}s")
    sorting = []

    # paper does it for first n/2 nodes but we can 
    # go all the way to n cuz we're too good at this game
    for l in range(n):
        if l == 0:
            # start = perf_counter()
            min_node = find_min(graph)
            # print(f"\tFound first node: {t(start)}s")
        else:
            # start = perf_counter()
            min_node = find(graph, min_node, levels, sorting, q)
            # print(f"\tFound next node ({l+1}): {t(start)}s")

        sorting.append(min_node)
        start = perf_counter()
        increment(graph, min_node, levels, elim)
        # print(f"\tIncrement on iter {l+1}: {t(start)}s")
        # # min here is to not get out of bounds issue
        # levels = create_level(edge_sets, levels, elim, min(q-1, update(l, p)), c)  
    
    return sorting
