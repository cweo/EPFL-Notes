# ThreeWayHikingTrails

## Description

- time limit per test: 15 seconds
- memory limit per test: 256 megabytes
- input: standard input
- output: standard output

It's the second year of your job as the Swiss Hiking Minister. During the past year, you managed to complete the first phase of an ambitious new project - a network of Three-way Hiking Trails! Each 3-way hiking trail connects three different mountain huts, and you have *m* such trails and *n* mountain huts numbered 1 to *n*.

In the second phase of the project, you need to ensure that every trail is *reachable*. That is, at least one of the three mountain huts that it connects must be upgraded to become a train station. It costs $c_i$ CHF to upgrade the *i*-th hut to a station. You would like to select a set of huts to upgrade whose total cost is as low as possible.

As this sounds like a difficult challenge, you do not need to compute the best possible solution. However, taxpayer money cannot be wasted: your solution must be at most *three* times more expensive than the cheapest one.

For reasons of transparency, you will need to provide evidence that this is the case. You will do so by assigning a nonnegative integer *budget* to each trail. These budgets must satisfy two conditions. First, the cost of your solution must stay within three times the budget, i.e., the sum of hut upgrade costs must be at most three times the sum of budgets of all trails. Second, for each hut (regardless of whether it will be upgraded), the sum of budgets of trails connected to it must be at most the upgrade cost for that hut.

If you cannot do it, the federal council will not approve your budget, and the whole project will be in jeopardy. Are you ready to save your precious project?

## Input and output format

### Input

The first line of the input contains two space-separated integers *n* and *m* – the number of huts and the number of trails, respectively (2 ≤ *n* ≤ 200, 1 ≤ *m* ≤ 4000).

The second line contains *n* space-separated integers $c_1, c_2, ..., c_n$, where $c_i$ is the cost of upgrading the *i*-th hut to a train station (1 ≤ $c_i$ ≤ 106).

The following *m* lines describe the trails. The *i*-th of these lines describes the *i*-th trail and contains three space-separated integers *u*, *v*and *w*. (1 ≤ *u* < *v* < *w* ≤ *n*) that describe a three-way trail connecting the mountain huts numbered *u*, *v*, and *w*. The trails do not repeat.

### Output

Your program must print three lines.

On the first line, output the number *k* of huts in your solution (i.e., those that you will upgrade to train stations).

On the second line, output *k* space-separated integers $v_1, v_2, ..., v_k$ – the numbers of these huts (1 ≤ $v_i$≤ *n*). They can be printed in any order, but should not repeat.

On the third line, print *m* space-separated integers $b_1, b_2, ..., b_m$ – the budgets that you have assigned to the trails, in the order in which the trails appear in the input (0 ≤ $b_j$ ≤ 109).

If there are many possible correct solutions, you may output any of them.

## Examples

input

```
5 5
10 2 6 3 7
1 2 3
1 2 4
1 2 5
2 4 5
3 4 5
```

output

```
2
2 5
0 0 2 0 3
```

input

```
5 10
10 80 40 20 5
1 2 3
1 2 4
1 2 5
1 3 4
1 3 5
1 4 5
2 3 4
2 3 5
2 4 5
3 4 5
```

output

```
3
1 4 5 
10 0 0 0 0 0 20 5 0 0 
```

Note

In the first example test case above, there are five huts and five trails. The cheapest solution would be to upgrade huts 2 and 4, for a cost of 2 + 3 = 5. However, upgrading huts 2 and 5 is also a good solution, with a cost of 2 + 7 = 9. Allocating a budget of 2 to trail (1, 2, 5)and a budget of 3 to trail (3, 4, 5) is feasible, the total budget is then 2 + 3 = 5, and we have 9 ≤ 3·5.