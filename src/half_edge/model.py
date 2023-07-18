from typing import List, Union
import numpy as np 
from open3d.utility import  Vector3dVector, Vector3iVector
from open3d.geometry import HalfEdge, TriangleMesh, HalfEdgeTriangleMesh
import vector_tools 

class HalfEdgeModel:

    def __init__(self, vertices: Union[np.ndarray, Vector3dVector], triangles: Union[np.ndarray, Vector3iVector]) -> None:

        self.unreferenced_vertices = []
        self.unreferenced_triangles = []
        self.unreferenced_half_edges = []

        self.n_vertices = vertices.shape[0] if isinstance(vertices, np.ndarray) else np.asarray(vertices).shape[0]
        vertices = Vector3dVector(vertices) if isinstance(vertices, np.ndarray) else vertices
        triangles = Vector3iVector(triangles) if isinstance(triangles, np.ndarray) else triangles

        _triangle_mesh = TriangleMesh(vertices, triangles)
        self._model = HalfEdgeTriangleMesh.create_from_triangle_mesh(_triangle_mesh)

        self.half_edges = self._model.half_edges
        self.vertices = self._model.vertices
        self.triangles = self._model.triangles

    def clean(self):
        _triangle_mesh = TriangleMesh(self.vertices, self.triangles)
        _triangle_mesh.remove_triangles_by_index(self.unreferenced_triangles)
        _triangle_mesh.remove_vertices_by_index(self.unreferenced_vertices)

        self.vertices = _triangle_mesh.vertices
        self.triangles = _triangle_mesh.triangles
        self.half_edges = [ h for i, h in enumerate(self.half_edges) if i not in self.unreferenced_half_edges ]

        self.unreferenced_vertices = []
        self.unreferenced_triangles = []
        self.unreferenced_half_edges = []

    def topology_checker(self, clean=True):
        if clean: self.clean()
        _triangle_mesh = TriangleMesh(self.vertices, self.triangles)
        print('===== TOPOLOGY CHECKER =====')
        print(f'Euler characteristic = {_triangle_mesh.euler_poincare_characteristic()}')
        print(f'Watertight = {_triangle_mesh.is_watertight()}')
        print(f'Orientable = {_triangle_mesh.is_orientable()}')
        print(f'Vertices start/end = {self.n_vertices}/{self.amount_of_vertices()}')
        print(f'Vertex non-manifold = {np.asarray(_triangle_mesh.get_non_manifold_vertices()).shape[0]}')
        print(f'Edge non-manifold = {np.asarray(_triangle_mesh.get_non_manifold_edges()).shape[0]}')
        print(f'Self-intersection = {np.asarray(_triangle_mesh.get_self_intersecting_triangles()).shape[0]}')

    # basic shape methods

    def amount_of_vertices(self):
        return np.asarray(self.vertices).shape[0]
    
    def amount_of_triangles(self):
        return np.asarray(self.triangles).shape[0]
    
    def amount_of_half_edges(self):
        return len(self.half_edges)

    # create/update half edge methods 

    def create_half_edge(self, next:int=-1, triangle_index:int=-1, twin:int=-1, vertex_indices:list=[-1, -1]):
        h = HalfEdge()
        h.next = next
        h.triangle_index = triangle_index
        h.twin = twin
        h.vertex_indices = vertex_indices
        return h
    
    def update_half_edge(self, h_index:int, next:int=None, triangle_index:int=None, twin:int=None, vertex_indices:list=None):
        h = self.half_edges[h_index]
        if next is not None: h.next = next
        if triangle_index is not None: h.triangle_index = triangle_index
        if twin is not None: h.twin = twin
        if vertex_indices is not None: h.vertex_indices = vertex_indices

    # get half edge data methods

    def get_twin_index(self, h_index:int):
        return self.half_edges[h_index].twin

    def get_next_index(self, h_index:int):
        return self.half_edges[h_index].next

    def get_triangle_index(self, h_index:int):
        return self.half_edges[h_index].triangle_index

    def get_vertex_indices(self, h_index:int):
        return self.half_edges[h_index].vertex_indices
    
    def get_end_vertex_index(self, h_index:int):
        return self.get_vertex_indices(h_index)[1]
    
    def get_start_vertex_index(self, h_index:int):
        return self.get_vertex_indices(h_index)[0]

    # get vertices and triangles methods

    def get_vertices_by_indices(self, indices: Union[int, List[int]]):
        return np.asarray(self.vertices)[indices] # .copy()
    
    def get_triangles_by_indices(self, indices: Union[int, List[int]]):
        return np.asarray(self.triangles)[indices] # .copy()

    def get_vertices_by_edge(self, h_index:int):
        indices = self.get_vertex_indices(h_index)
        return self.get_vertices_by_indices(indices)
    
    def get_start_vertex_by_edge(self, h_index:int):
        v0_index = self.get_start_vertex_index(h_index)
        return self.get_vertices_by_indices(v0_index)
    
    def get_end_vertex_by_edge(self, h_index:int):
        v1_index = self.get_end_vertex_index(h_index)
        return self.get_vertices_by_indices(v1_index)
    
    def get_triangle_indices_by_edge(self, h_index:int):
        t_index = self.get_triangle_index(h_index)
        return self.get_triangles_by_indices(t_index)

    def get_triangle_vertices_by_edge(self, h_index:int):
        triangle = self.get_triangle_indices_by_edge(h_index)
        return self.get_vertices_by_indices(triangle)
    
    # adjacent methods

    def adjacent_half_edges(self, h_index:int):
        nh_index = self.get_next_index(h_index)
        nnh_index = self.get_next_index(nh_index)
        th_index = self.get_twin_index(h_index)
        nth_index = self.get_next_index(th_index)
        nnth_index = self.get_next_index(nth_index)
        return nh_index, nnh_index, nth_index, nnth_index

    def adjacent_triangles(self, h_index: int):
        f0_index = self.get_triangle_index(h_index)
        th_index = self.get_twin_index(h_index)
        f1_index = self.get_triangle_index(th_index)
        return f0_index, f1_index
    
    # computing methods

    def edge_len(self, h_index:int):
        vertices = self.get_vertices_by_edge(h_index)
        return vector_tools.distance(vertices)

    def valence(self, h_index:int):
        return len(list(self.edge_ring(h_index)))

    def triangle_normal(self, h_index:int):
        vertices = self.get_triangle_vertices_by_edge(h_index)
        return vector_tools.triangle_normal(vertices, normalize=True)

    def vertex_normal(self, h_index:int): 
        r = list(self.normal_ring(h_index))
        return sum(r)/len(r)
    
    def mean_vertex(self, h_index:int): 
        indices = list(self.vertex_ring(h_index))
        vertices = self.get_vertices_by_indices(indices)
        return np.mean(vertices, axis=0)
    
    def compactness(self, h_index:int):
        vertices = self.get_triangle_vertices_by_edge(h_index)
        return vector_tools.triangle_compactness(vertices)

    def mean_compactness(self, h_index:int):
        r = list(self.compactness_ring(h_index))
        return sum(r)/len(r)
    
    # ring methods

    def edge_ring(self, h_index:int):
        # argument of fun must be a h_index
        last = h_index
        # Clockwise edge index return
        while True:
            th_index = self.get_twin_index(h_index)
            nth_index = self.get_next_index(th_index)
            yield nth_index
            if nth_index == last:
                break
            h_index = nth_index
    
    def vertex_ring(self, h_index:int):
        return map(self.get_end_vertex_index, self.edge_ring(h_index))

    def triangle_ring(self, h_index:int):
        return map(self.get_triangle_index, self.edge_ring(h_index))

    def normal_ring(self, h_index:int):
        return map(self.triangle_normal, self.edge_ring(h_index))

    def compactness_ring(self, h_index:int):
        return map(self.compactness, self.edge_ring(h_index))

    # set vertices and triangles methods

    def replace_vertex_by_index(self, index: int, new_vertex: np.ndarray):
        self.vertices.insert(index, new_vertex)
        self.vertices.pop(index+1)

    def replace_triangle_by_index(self, index: int, new_triangle: np.ndarray):
        self.triangles.insert(index, new_triangle)
        self.triangles.pop(index+1)

    def update_triangle_by_vertex_indices(self, h_index:int, v_out:int, v_target:int):
        t_index = self.get_triangle_index(h_index)
        triangle = self.get_triangles_by_indices(t_index)
        triangle[triangle==v_target] = v_out
        self.replace_triangle_by_index(t_index, triangle)

    # half edge basic operations 

    def edge_split(self, h0_index:int):

        E = self.amount_of_half_edges()
        T = self.amount_of_triangles()
        V = self.amount_of_vertices()

        # get data

        h1_index = self.get_next_index(h0_index)
        h2_index = self.get_next_index(h1_index)
        h3_index = self.get_twin_index(h0_index)

        h4_index = self.get_next_index(h3_index)
        h5_index = self.get_next_index(h4_index)

        t0_index = self.get_triangle_index(h0_index)
        t1_index = self.get_triangle_index(h3_index)

        v0_index, v1_index = self.get_vertex_indices(h4_index)
        v2_index, v3_index = self.get_vertex_indices(h1_index)

        # create new half edges indices

        h6_index = E+0
        h7_index = E+1
        h8_index = E+2
        h9_index = E+3
        h10_index = E+4
        h11_index = E+5
        h12_index = E+6
        h13_index = E+7

        # create new triangles indices

        t2_index = T+0
        t3_index = T+1
        t4_index = T+2
        t5_index = T+3

        # create new vertex index

        v4_index = V+0

        # create new half edges

        h6 = self.create_half_edge(
            next=h7_index,
            triangle_index=t2_index,
            twin=h8_index,
            vertex_indices=[v0_index, v4_index]
        )

        h7 = self.create_half_edge(
            next=h2_index,
            triangle_index=t2_index,
            twin=h12_index,
            vertex_indices=[v4_index, v3_index]
        )

        h8 = self.create_half_edge(
            next=h4_index,
            triangle_index=t3_index,
            twin=h6_index,
            vertex_indices=[v4_index, v0_index]
        )

        h9 = self.create_half_edge(
            next=h8_index,
            triangle_index=t3_index,
            twin=h10_index,
            vertex_indices=[v1_index, v4_index]
        )

        h10 = self.create_half_edge(
            next=h5_index,
            triangle_index=t4_index,
            twin=h9_index,
            vertex_indices=[v4_index, v1_index]
        )

        h11 = self.create_half_edge(
            next=h10_index,
            triangle_index=t4_index,
            twin=h13_index,
            vertex_indices=[v2_index, v4_index]
        )

        h12 = self.create_half_edge(
            next=h13_index,
            triangle_index=t5_index,
            twin=h7_index,
            vertex_indices=[v3_index, v4_index]
        )

        h13 = self.create_half_edge(
            next=h1_index,
            triangle_index=t5_index,
            twin=h11_index,
            vertex_indices=[v4_index, v2_index]
        )


        # create new triangles
        t2 = np.array([v4_index, v3_index, v0_index])
        t3 = np.array([v4_index, v0_index, v1_index])
        t4 = np.array([v4_index, v1_index, v2_index])
        t5 = np.array([v4_index, v2_index, v3_index])

        # create new vertex
        vertices = self.get_vertices_by_indices([v0_index, v2_index])
        v4 = vector_tools.midpoint(vertices)

        # update half edges
        self.update_half_edge(
            h1_index,
            next=h12_index,
            triangle_index=t5_index
        )

        self.update_half_edge(
            h2_index,
            next=h6_index,
            triangle_index=t2_index
        )

        self.update_half_edge(
            h4_index,
            next=h9_index,
            triangle_index=t3_index
        )

        self.update_half_edge(
            h5_index,
            next=h11_index,
            triangle_index=t4_index
        )

        # insert new half edges
        self.half_edges += [h6, h7, h8, h9, h10, h11, h12, h13] # caution: extend() does not work.

        # insert new triangles
        self.triangles.extend([t2, t3, t4, t5])

        # insert new vertex
        self.vertices.append(v4)

        # save unreferenced half edges
        self.unreferenced_half_edges.extend([h0_index, h3_index])

        # save unreferenced triangles
        self.unreferenced_triangles.extend([t0_index, t1_index])

    def revert_edge_collapse(self, p_ring:list):

        # get data 

        h5_index = self.unreferenced_half_edges[-1]
        h4_index = self.unreferenced_half_edges[-2]
        h2_index = self.unreferenced_half_edges[-4]
        h1_index = self.unreferenced_half_edges[-5]

        h6_index = self.get_twin_index(h2_index)
        h7_index = self.get_twin_index(h4_index)
        h8_index = self.get_twin_index(h5_index)
        h9_index = self.get_twin_index(h1_index)

        v0_index, v3_index = self.get_vertex_indices(h4_index)
        v1_index, v2_index = self.get_vertex_indices(h1_index)

        # updates 

        for h_index in p_ring:

            _, v_index = self.get_vertex_indices(h_index)
            
            # update triangle 
            self.update_triangle_by_vertex_indices(h_index, v1_index, v0_index)

            if v_index == v3_index:
                continue

            # update half edge vertices

            self.update_half_edge(
                h_index,
                vertex_indices=[v1_index, v_index]
            )

            th_index = self.get_twin_index(h_index)
            self.update_half_edge(
                th_index,
                vertex_indices=[v_index, v1_index]
            )

        # update half edge twins

        self.update_half_edge(
            h6_index,
            twin=h2_index
        )

        self.update_half_edge(
            h9_index,
            twin=h1_index,
            vertex_indices=[v2_index, v1_index]
        )

        self.update_half_edge(
            h7_index,
            twin=h4_index
        )

        self.update_half_edge(
            h8_index,
            twin=h5_index,
            vertex_indices=[v1_index, v3_index]
        )

        # remove unreferenced half edges
        self.unreferenced_half_edges = self.unreferenced_half_edges[:-6]

        # remove unreferenced triangles
        self.unreferenced_triangles = self.unreferenced_triangles[:-2]

        # remove unreferenced vertices
        self.unreferenced_vertices.pop()

    def edge_flip(self, h0_index:int):

        # get data

        h1_index = self.get_next_index(h0_index)
        h2_index = self.get_next_index(h1_index)
        h3_index = self.get_twin_index(h0_index)

        h4_index = self.get_next_index(h3_index)
        h5_index = self.get_next_index(h4_index)

        h6_index = self.get_twin_index(h1_index)
        h7_index = self.get_twin_index(h2_index)
        h8_index = self.get_twin_index(h4_index)
        h9_index = self.get_twin_index(h5_index)

        t0_index = self.get_triangle_index(h0_index)
        t1_index = self.get_triangle_index(h3_index)

        v2_index, v3_index = self.get_vertex_indices(h1_index)
        v0_index, v1_index = self.get_vertex_indices(h4_index)

        # Update half edges

        self.update_half_edge(
            h0_index,
            vertex_indices=[v3_index, v1_index]
        )

        self.update_half_edge(
            h1_index,
            twin=h9_index,
            vertex_indices=[v1_index, v2_index]
        )

        self.update_half_edge(
            h2_index,
            twin=h6_index,
            vertex_indices=[v2_index, v3_index]
        )

        self.update_half_edge(
            h3_index,
            vertex_indices=[v1_index, v3_index]
        )

        self.update_half_edge(
            h4_index,
            twin=h7_index,
            vertex_indices=[v3_index, v0_index]
        )

        self.update_half_edge(
            h5_index,
            twin=h8_index,
            vertex_indices=[v0_index, v1_index]
        )

        self.update_half_edge(
            h6_index,
            twin=h2_index
        )

        self.update_half_edge(
            h7_index,
            twin=h4_index
        )

        self.update_half_edge(
            h8_index,
            twin=h5_index
        )

        self.update_half_edge(
            h9_index,
            twin=h1_index
        )

        # Update triangles
        self.replace_triangle_by_index(t0_index, np.array([v1_index, v2_index, v3_index]))
        self.replace_triangle_by_index(t1_index, np.array([v0_index, v1_index, v3_index]))

    def edge_collapse(self, h0_index: int) -> list:
        
        # get data 

        h1_index = self.get_next_index(h0_index)
        h2_index = self.get_next_index(h1_index)
        h3_index = self.get_twin_index(h0_index)

        h4_index = self.get_next_index(h3_index)
        h5_index = self.get_next_index(h4_index)

        h6_index = self.get_twin_index(h2_index)
        h7_index = self.get_twin_index(h4_index)
        h8_index = self.get_twin_index(h5_index)
        h9_index = self.get_twin_index(h1_index)

        t0_index = self.get_triangle_index(h0_index)
        t1_index = self.get_triangle_index(h3_index)

        v0_index, v3_index = self.get_vertex_indices(h4_index)
        v1_index, v2_index = self.get_vertex_indices(h1_index)

        # updates 

        v1_ring = self.edge_ring(h3_index)
        p_ring = []

        for h_index in v1_ring:

            _, v_index = self.get_vertex_indices(h_index)

            if v_index in (v0_index, v2_index):
                continue

            # update triangle 

            self.update_triangle_by_vertex_indices(h_index, v0_index, v1_index)

            p_ring.append(h_index) 

            if v_index == v3_index:
                continue

            # update half edge vertices

            self.update_half_edge(
                h_index,
                vertex_indices=[v0_index, v_index]
            )

            th_index = self.get_twin_index(h_index)
            self.update_half_edge(
                th_index,
                vertex_indices=[v_index, v0_index]
            )

        # update half edge twins

        self.update_half_edge(
            h6_index,
            twin=h9_index
        )

        self.update_half_edge(
            h9_index,
            twin=h6_index,
            vertex_indices=[v2_index, v0_index]
        )

        self.update_half_edge(
            h7_index,
            twin=h8_index
        )

        self.update_half_edge(
            h8_index,
            twin=h7_index,
            vertex_indices=[v0_index, v3_index]
        )

        # save unreferenced half edges
        self.unreferenced_half_edges.extend([h0_index, h1_index, h2_index, h3_index, h4_index, h5_index])

        # save unreferenced triangles
        self.unreferenced_triangles.extend([t0_index, t1_index])

        # save unreferenced vertex
        self.unreferenced_vertices.extend([v1_index])

        return p_ring 