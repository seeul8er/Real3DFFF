##Copyright 2008-2015 Jelle Feringa (jelleferinga@gmail.com)
##
##This file is part of pythonOCC.
##
##pythonOCC is free software: you can redistribute it and/or modify
##it under the terms of the GNU Lesser General Public License as published by
##the Free Software Foundation, either version 3 of the License, or
##(at your option) any later version.
##
##pythonOCC is distributed in the hope that it will be useful,
##but WITHOUT ANY WARRANTY; without even the implied warranty of
##MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##GNU Lesser General Public License for more details.
##
##You should have received a copy of the GNU Lesser General Public License
##along with pythonOCC.  If not, see <http://www.gnu.org/licenses/>.

'''
This module helps looping through topology
'''
from OCC.BRep import BRep_Tool

from OCCUtils.Topology import WireExplorer, Topo
from OCCUtils.edge import Edge


class EdgePairsFromWire(object):
    '''
    helper class to loop through a wire and return ordered pairs of edges
    '''
    def __init__(self, wire):
        self.wire = wire
        self.edge_pairs = []
        self.prev_edge = None
        self.we = WireExplorer(self.wire).ordered_edges()
        self.number_of_edges = self.we.__length_hint__()
        self.previous_edge = None
        self.current_edge = None
        self.first_edge = None
        self.index = 0

    def next(self):
        if self.index == 0:
            # first edge, need to set self.previous_edge
            self.previous_edge = next(self.we)
            self.current_edge = next(self.we)
            self.first_edge = self.previous_edge   # for the last iteration
            self.index += 1
            return [self.previous_edge, self.current_edge]
        elif self.index == self.number_of_edges-1:
            # no next edge
            self.index += 1
            return [self.current_edge, self.first_edge]
        else:
            self.previous_edge = self.current_edge
            self.current_edge = next(self.we)
            self.index += 1
            return [self.previous_edge, self.current_edge]

    def __iter__(self):
        return self


class LoopWirePairs(object):
    '''
    for looping through consequtive wires
    assures that the returned edge pairs are ordered
    '''
    def __init__(self, wireA, wireB):
        self.wireA = wireA
        self.wireB = wireB
        self.we_A = WireExplorer(self.wireA)
        self.we_B = WireExplorer(self.wireB)
        self.tp_A = Topo(self.wireA)
        self.tp_B = Topo(self.wireB)
        self.bt = BRep_Tool()
        self.vertsA = [v for v in self.we_A.ordered_vertices()]
        self.vertsB = [v for v in self.we_B.ordered_vertices()]

        self.edgesA = [v for v in WireExplorer(wireA).ordered_edges()]
        self.edgesB = [v for v in WireExplorer(wireB).ordered_edges()]

        self.pntsB = [self.bt.Pnt(v) for v in self.vertsB]
        self.number_of_vertices = len(self.vertsA)
        self.index = 0

    def closest_point(self, vertexFromWireA):
        pt = self.bt.Pnt(vertexFromWireA)
        distances = [pt.Distance(i) for i in self.pntsB]
        indx_max_dist = distances.index(min(distances))
        return self.vertsB[indx_max_dist]

    def next(self):
        if self.index == self.number_of_vertices:
            raise StopIteration

        vert = self.vertsA[self.index]
        closest = self.closest_point(vert)
        edges_a = self.tp_A.edges_from_vertex(vert)
        edges_b = self.tp_B.edges_from_vertex(closest)
        a1, a2 = Edge(next(edges_a)), Edge(next(edges_a))
        b1, b2 = Edge(next(edges_b)), Edge(next(edges_b))
        mpA = a1.mid_point()
        self.index += 1

        if mpA.Distance(b1.mid_point()) < mpA.Distance(b2.mid_point()):
            return iter([a1, a2]), iter([b1, b2])
        else:
            return iter([a1, a2]), iter([b2, b1])

    def __iter__(self):
        return self
