import torch


def projection(vertices, K, orig_size_y, orig_size_x):
    '''
    Calculate projective transformation of vertices given a projection matrix
    Input parameters:
    K: batch_size * 3 * 3 intrinsic camera matrix
    R, t: batch_size * 3 * 3, batch_size * 1 * 3 extrinsic calibration parameters
    orig_size_y: original height of image captured by the camera
    orig_size_x: original width of image captured by the camera
    Returns: For each point [X,Y,Z] in world coordinates
      [u,v,z] where u,v are the coordinates of the projection in
    pixels and z is the depth
    '''

    # instead of P*x we compute x'*P'
    vertices = torch.mm(vertices, K.transpose(1, 0))

    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    x_ = x / z
    y_ = y / z

    x_ = 2 * (x_ - orig_size_x / 2.) / orig_size_x
    y_ = 2 * (y_ - orig_size_y / 2.) / orig_size_y
    vertices = torch.cat((x_.unsqueeze(1), y_.unsqueeze(1), z.unsqueeze(1)),
                         dim=1)
    return vertices
