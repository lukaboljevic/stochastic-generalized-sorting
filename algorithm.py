"""
Notes

- Construct edge sets
- Assign vertices to levels according to a certain promotion rule
- x_l is the most recent vertex 
- Levels have to be updated all the time
- Levels are periodically rebuilt from scratch - when l = k*(s_i/32)
    where s_i = 2^i / p for level Li and probability p
    refer to s_i as the TARGET SIZE for level Li
- candidate set for finding next vertex x_l+1
"""

"""
We have G = (V, E), with probability p, x1 ≺ x2 ≺ ... ≺ xn is the true order

edge (u, v) is STOCHASTIC if it truly is the random edge
edge (u, v) is DETERMINISTIC if it is was manually included in the graph
    i.e. it's an edge of the true ordering path

vertices are found one at a time
when a vertex is found, we say it's DISCOVERED
    as convention, define xl to be the most recent discovered vertex

rank of a vertex - its rank out of the not yet discovered vertices
{xl+1, ..., xn}
    hence, for a vertex v, r(v) is the rank, so v = x(l+r(v))

the algorithm only considers the task of finding x1, ... x(n/2).
With a symmetric argument, xn, ..., x(n/2+1) can be found
"""

"""
#### CONSTRUCTING THE EDGE SETS ####

Edge sets E1, E2, ... Eq where q = O(lg(pn))
    these sets are made once and NEVER modified.

Ei(u, v) - indicator random variable for whether (u, v) in Ei
E(u, v) - tuple <E1(u, v), E2(u, v), ..., Eq(u, v)>

these sets are not necessarily disjoint
"""

from math import log
# from sympy import symbols, solve
# from sympy.core.numbers import Float
from random import random, choice
from math import floor, log2
# import networkx as nx

# def calculate_alpha(p, q):
#     x = symbols("x") # "alpha"

#     prod = 1
#     for i in range (1, q+1):
#         prod *= (1 - (x*p) / (2 ** i))

#     prod -= (1 - p)
#     solution = solve(prod, x)
#     solution = filter(lambda elem: isinstance(elem, Float) and 
#         1 < elem and elem < 2, solution)
#     return float(list(solution)[0])

# def edge_sets_sample_probability(p, q, alpha):
    # l = []
    # while True:
    #     for i in range(q):
    #         l.append(1 if random() < alpha*p / 2 **(i+1) else 0)

    #     if l.count(1) > 0:
    #         break
    #     else:
    #         l = []
    
    # return l.count(1) / q # q == len(l)
    # return 0.6

def query(u, v):
    return u < v

def construct_edge_sets(n, edges, q, prob):
    """
    If you read the paper, where it explains how to make these edge sets
    it needs to solve for some variable "alpha", which is my parameter 
    "prob" here. I implemented solving for this alpha above, using sympy,
    but I just fixed some probability to avoid complications for now, as
    I didn't fully understand it - other stuff is more important
    """
    # prob = edge_sets_sample_probability(p, q, calculate_alpha(p, q))
    necessary_edges = set()
    for i in range(n-1):
        # since it will rarely happen that I will add all edges to all levels,
        # I say "okay", these are the ones you have to have in the levels at the
        # very least, since these make up the Hamiltonian path
        necessary_edges.add((i, i+1))
        necessary_edges.add((i+1, i))
    while True:
        E = {i: [] for i in range(q)}
        s = set()
        for edge in edges:
            for num, level in E.items():
                if random() < prob:
                    level.append(edge)
                    s.add(edge)
        # should be =, or at the very least, ask
        # if at least the edges from the Ham. path have been included
        # idk if this is good
        if len(s) > 0.9*len(edges) and len(necessary_edges.intersection(s)) == len(necessary_edges):
            return E

def find_min(nodes, edges):
    """
    "Directly" from paper
    """
    min_elem = None
    s = set()
    edge = choice(edges)
    u, v = edge
    if query(u, v):
        min_elem = u
        s.add(v)
    else:
        min_elem = v
        s.add(u)
    
    # this loop can be optimized looking at the paper 
    # but this function at least works properly xD
    for u in nodes:
        if (u, min_elem) in edges and u not in s:
            if query(u, min_elem):
                s.add(min_elem)
                min_elem = u
            else:
                s.add(u)
    return min_elem

def create_level(edge_sets, levels, elim, i, c):
    """
    "Directly" from paper
    """
    while i >= 0:
        levels[i] = levels[i+1].copy()
        for v in levels[i+1]:
            elim[v] = None
            for u in levels[min(i+c, len(levels)-1)]: # min is here so that we don't go out of bounds
                if (u, v) in edge_sets[i] and query(u, v):
                    if v in levels[i]:
                        levels[i].remove(v)
                    elim[v] = u
        
        i -= 1
    return levels

def lowest_level_containing(node, levels):
    for i, level in enumerate(levels):
        if node in level:
            return i

def increment(last_min, nodes, levels, edge_sets, elim, c):
    """
    More or less "directly" from paper
    """
    for level in levels:
        if last_min in level:
            level.remove(last_min)

    for v in nodes:
        if elim[v] != last_min:
            continue
        elim[v] = None
        i = lowest_level_containing(v, levels)
        i = min(i, len(edge_sets) - 1) # otherwise it goes out of bounds
        while i > 1:
            for u in levels[min(i+c, len(levels)-1)]: # otherwise out of bounds problem
                if (u, v) in edge_sets[i]:
                    if query(u, v):
                        elim[v] = u
                        break
            i = i - 1
            levels[i-1].append(v)
        
def find(remaining_nodes, edges, previous_min, q, levels):
    """
    If this function could perhaps work, maybe we'd have the
    algorithm - at the very least it would do something
    """
    s = set()
    for v in remaining_nodes:
        # remaining_nodes cuz it says "for each v not in {x1, ..., x_l-1}"
        if v in levels[0] and (previous_min, v) in edges:
            s.add(v)

    for i in range(q):
        for v in s:
            for u in levels[i]:
                if (u, v) in edges:
                    if query(u, v):
                        s.remove(v)
                print(len(s))
                if len(s) == 1:
                    for elem in s: # select the only element
                        break
                    return elem
    # this function USUALLY returns none and that bothers me

def highest_power_of_2(n):
    # found this online, don't ask me xD
    # it's for the 1 + "largest exponent of 2 dividing l'" from the paper
    return int(log2((n & (~(n - 1))))) 

def update(l, p):
    x = l + 1
    while True:
        if l == floor(x / (16 * p)):
            break
        x = x + 1
    return 1 + highest_power_of_2(x)

def stochastic_sort(nodes, edges, p, q, c, edge_set_prob):
    # Set up
    levels = []
    levels.append([])
    n = len(nodes)
    for i in range(q + c): # this many levels
        if i < q-1:
            levels.append([]) # the idea is to construct the lower levels from the top ones
        else:
            levels.append(nodes) # the topmost levels have all the nodes
    elim = [None] * len(nodes)
    edge_sets = construct_edge_sets(n, edges, q, edge_set_prob) # construct the edge sets
    levels = create_level(edge_sets, levels, elim, q-1, c) # create bottom levels
    sorting = []
    remaining_nodes = nodes.copy()

    for l in range(round(n/2)):
        if l == 0:
            min_node = find_min(nodes, edges)
        else:
            # this is sadly always none =(
            min_node = find(remaining_nodes, edges, min_node, q, levels)

        sorting.append(min_node)
        if min_node in remaining_nodes:
            remaining_nodes.remove(min_node)

        increment(min_node, nodes, levels, edge_sets, elim, c)
        # min here is to not get out of bounds issue
        levels = create_level(edge_sets, levels, elim, min(q-1, update(l, p)), c)  
    
    return sorting

def order(i, j):
    return (i, j) if i < j else (j, i)

def create_graph(n, p):
    """
    Create a random graph with a Hamiltonian path
    THIS FUNCTION ACTUALLY WORKS!!!! unlike others -_-
    """
    nodes = list(range(n))
    edges = []
    for i in range(n-1):
        # Include the Hamiltonian path
        edges.append((i, i+1))
        edges.append((i+1, i))

    for i in range(n-1):
        for j in range(1, n):
            if i != j and order(i, j) not in edges and random() < p:
                edges.append((i, j))
                edges.append((j, i))
    
    return nodes, edges


n = 100
p = 0.4
q = round(log(p*n)) # paper said q = O(log(np)) B)
nodes, edges = create_graph(n, p)
# import networkx as nx
# g = nx.Graph()
# g.add_nodes_from(nodes)
# g.add_edges_from(edges)
# first_half = g.subgraph(nodes[0:round(n/2)])
# first_half_nodes = list(g.nodes())
# first_half_edges = list(g.edges())
# second_half = g.subgraph(nodes[round(n/2)+1:])
# nodes = [0, 1, 2, 3, 4, 5]
# edges = [(0, 1), (1, 0), 
#     (1, 2), (2, 1),
#     (0, 2), (2, 0),
#     (0, 3), (3, 0),
#     (3, 5), (5, 3),
#     (4, 5), (5, 4),
#     (4, 1), (1, 4)]

"""
Beware: the algorithm sorts half the nodes to begin with!
but it doesn't work B)
"""
result = stochastic_sort(nodes, edges, p, q, 10, 0.7)
print(result)

# prob = 0.5
# E = construct_edge_sets(edges, q, 0.5)
# print(E)