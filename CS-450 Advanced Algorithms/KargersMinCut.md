# Karger’s min-cut algorithm

## Description

- time limit per test: 15 seconds
- memory limit per test: 256 megabytes
- input: standard input
- output: standard output

You are given a connected undirected graph (with possibly multiple edges between the same pair of vertices). Using Karger's min-cut algorithm, find and return the size of the minimum cut in the graph, as well as the number of minimum cuts.

## Input and output format

Input

The first line of the input contains two space-separated integers *n* and *m* (2 ≤ *n* ≤ 100, 1 ≤ *m* ≤ 400) – the number of vertices and the number of edges, respectively. The next *m* lines describe the edges. Each such line contains two space-separated integers *a* and *b*(1 ≤ *a*, *b* ≤ *n*, *a* ≠ *b*) – endpoints of the edge. There may be multiple edges between the same pair of vertices. The graph is connected.

Output

Output two integers – the size of the minimum cut and the number of cuts of this size in the graph. Two cuts are different if the corresponding sets of edges are different.

## Examples

input

```
5 5
1 2
2 3
3 4
4 5
5 1
```

output

```
2 10
```

input

```
2 1
1 2
```

output

```
1 1
```

Note

The first example testcase is a 5-cycle. The graph can be cut by removing any two edges; on the other hand, removing any single edge does not disconnect the graph. Therefore the size of the min cut is 2, and there are ${5}\choose{2}$min cuts.

For your probability estimations, you may find it useful to know that there are at most 100 test cases on which your program will be run.