import torch


def vertices_to_faces(vertices, faces):
    """
    :param vertices: [number of vertices, 3]
    :param faces: [number of faces, 3)
    :return: [number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 2)
    assert (faces.ndimension() == 2)
    # assert (vertices.shape[0] == faces.shape[0])
    # assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    nv = vertices.shape[0]
    device = vertices.device

    return vertices[faces.long()]
