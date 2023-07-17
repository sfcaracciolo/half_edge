# Half-Edge (data structure)

An implementation of half-edge data structure based on ```open3d.geometry.HalfEdgeTriangleMesh``` adding this operations:

* **edge-split**: split an edge on the midpoint
* **edge-flip**: switch the vertices in contact with an edge to the vertices non-shared of the adjacent triangles
* **edge-collapse**: collapse an edge reconnecting all triangles to the corresponding vertex.
* **revert edge-collapse**: undo the edge collapse operation.

## Usage

```python
from half_edge import HalfEdgeModel

vertices = # numpy.ndarray [float32] or open3d.utility.Vector3dVector
triangles = # numpy.ndarray [int32] or open3d.utility.Vector3iVector

model = HalfEdgeModel(vertices, triangles)

```

In ```model.half_edges``` is stored a *list* with the half edges. You can use a index *h_index* of any half edge to operate on it,

```python
model.split_edge(h_index)
model.edge_flip(h_index)
p_ring = model.edge_collapse(h_index)
```

if you want revert the collpase operation,

```python
model.revert_edge_collapse(p_ring)
```

When operations are performed, vertices, triangles and half edges could be unreferenced, in this case you can track them on ```model.unreferenced_vertices```, ```model.unreferenced_triangles``` and ```model.unreferenced_half_edges```. 

In order to delete de unreferenced elements on ```model.vertices``` and ```model.triangles``` call ```model.clean()```.
