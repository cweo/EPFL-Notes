"""
Note that this is the weighted vertex cover problem on 3-uniform hypergraphs
(huts are vertices, trails are edges).
"""

import sys

lines = sys.stdin.readlines()

#define a dict:{vertex: all edges incident to the vertex}
incident_dict = {}

#preprocessing
for i, line in enumerate(lines):
    if i >= 2:
        lines[i] = list(map(lambda x: int(x)-1, line.strip("\n").split(" ")))
        for vertex in lines[i]:
            if vertex in incident_dict:
                incident_dict[vertex].append(i-2)
            else:
                incident_dict[vertex] = [i-2]
    else:
        lines[i] = list(map(lambda x: int(x), line.strip("\n").split(" ")))
        
nhubs = lines[0][0]
ntrails = lines[0][1]
totalcosts = lines[1]
trails = lines[2:]

#initialize to be all zeros
trail_budget = [0 for i in range(ntrails)]

#none of the trail is covered
trail_covered = [False for i in range(ntrails)]

#none of the hubs is chosen in C
hubs_chosen = []

for i in range(ntrails):
    #covered
    if trail_covered[i]:
        continue

    #not covered
    else:
        #get verttices of i
        vertices_left = list(set(trails[i]).difference(set(hubs_chosen)))
        costs = []
        for vert in vertices_left:
            #get all edges incident to vert
            edges = incident_dict[vert]
            costs.append(totalcosts[vert]-sum([trail_budget[i] for i in edges]))
        
        min_ind = costs.index(min(costs))
        chosen_vertex = vertices_left[min_ind]
        
        #increase budget
        trail_budget[i] += min(costs)

        #add vertex
        hubs_chosen.append(chosen_vertex)

        #update covered trail
        for trail_no in incident_dict[chosen_vertex]:
            trail_covered[trail_no] = True

#output         
print(len(hubs_chosen))
print(" ".join(list(map(lambda x: str(x+1), hubs_chosen))))
print(" ".join(list(map(lambda x: str(x), trail_budget))))