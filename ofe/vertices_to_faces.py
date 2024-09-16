import torch


def vertices_to_faces(vertices, faces):
    """
    :param vertices: [number of vertices, 3]
    :param faces: [number of faces, 3)
    :return: [number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 2)
    assert (faces.ndimension() == 2)
    assert (faces.shape[2] == 3)

    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]
