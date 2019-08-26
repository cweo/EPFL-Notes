import sys
import random

class Karger(object):
    
    def __init__(self, V, edges, min_cut):
        self.V = V
        self.edges = edges
        self.min_cut = min_cut
        self.min_cut_set = set()
        
    def _make_set(self):
        self._id = list(range(self.V))
        self._count = self.V
        self._rank = [0] * self.V
        
    def _find(self, i):
        while i != self._id[self._id[i]]:
            self._id[i] = i = self._id[self._id[i]]
        return i
    
    def _union(self, p, q):
        i, j = self._find(p), self._find(q)
        if i == j:
            return
        self._count -= 1
        if self._rank[i] < self._rank[j]:
            self._id[i] = j
        elif self._rank[i] > self._rank[j]:
            self._id[j] = i
        else:
            self._id[j] = i
            self._rank[i] += 1
            
    def _connected(self, i, j):
        return self._find(i) == self._find(j)
    
    def _iterate_contraction(self):
        #execute one iteration of edge contraction
        self._make_set()
        random.shuffle(self.edges)
        cut_size = 0
        
        for ind, (i, j) in enumerate(self.edges):
            if not self._connected(i, j):
                #haven't been contracted
                if self._count > 2:
                    self._union(i, j)
                else:
                    cut_size +=1
                    
        if cut_size > self.min_cut:
            #keep the old min cut
            return self.min_cut, self.min_cut_set
        
        elif cut_size < self.min_cut:
            self.min_cut = cut_size
            self.min_cut_set = set()
            
        self.min_cut_set.add(tuple([i for i in range(self.V) if self._connected(i, 0)]))
            
        return self.min_cut, self.min_cut_set
    
    def find_min_cut(self):
        for _ in range(min(14*self.V**2, 60000)):
            self.min_cut, self.min_cut_set = self._iterate_contraction()
        
        return self.min_cut, len(self.min_cut_set)

if __name__ == "__main__":
    random.seed(2)
    lines = sys.stdin.readlines()
    edges = []

    for i, line in enumerate(lines):
        if i >= 1:
            edges.append(tuple(map(lambda x: int(x)-1, line.strip("\n").split(" "))))
        else:
            V, E = map(lambda x: int(x), line.strip("\n").split(" "))
            
    k = Karger(V, edges, E)
    i, j = k.find_min_cut()    
    print i, j