from math import log
from algorithm import stochastic_sort, random_graph
from heapsort import heap_sort
from time import perf_counter

p = 0.1
for n in [10, 100, 1000]:
    print("=========================="*2)
    print(f"Size: {n}")
    q = max(2, round(log(p*n))) # paper said q = O(log(np)) B)
    print(q)
    graph = random_graph(n, p)
    nodes = list(graph.nodes)

    start = perf_counter()
    stochastic = stochastic_sort(graph, p, q)
    print(f"Stochastic sort time: {round(perf_counter() - start, 5)}s")

    start = perf_counter()
    heap_sort(nodes)
    print(f"Heap sort time: {round(perf_counter() - start, 5)}s")
