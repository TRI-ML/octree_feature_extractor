import torch as th
import torch.nn as nn

import ofe
import ofe.cuda.ofe as octree_feature_extractor_cuda


class OctreeFeatureExtractor(nn.Module):
    def __init__(self,
                 image_height=256,
                 image_width=256):
        super(OctreeFeatureExtractor, self).__init__()
        # rendering
        self.image_height = image_height
        self.image_width = image_width
        grid_offset = th.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]) - 0.5
        face_offset = th.tensor(
                [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 4, 5], [0, 5, 1],
                 [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [4, 0, 3], [4, 3, 7]])
        # self.register_buffer('K', K)
        self.register_buffer('grid_offset', grid_offset)
        self.register_buffer('face_offset', face_offset)

    def forward(self, pts, mask, depth_map, K, batch_id, grid_size):
        '''
        args:
            pts: (N, 3)
            mask: (B, H, W)
            depth_map: (B, H, W)
            K: (3, 3)
            batch_id: (N)
            grid_size: (1)
        return:
            octree_feature: (N, 2)
        '''
        num_voxels = pts.shape[0]
        device = pts.device

        # K = self.K.unsqueeze(0).repeat(bs, 1, 1)
        image_height = self.image_height
        image_width = self.image_width

        vertices = (pts.unsqueeze(1) + (grid_size * self.grid_offset).unsqueeze(0)).reshape(-1, 3)
        vertices = ofe.projection(vertices, K, image_height, image_width)
        faces = self.face_offset.repeat(num_voxels, 1) + th.arange(num_voxels, device=device).repeat_interleave(12).unsqueeze(-1) * 8
        face_vertices = ofe.vertices_to_faces(vertices, faces)
        octree_feature = octree_feature_extractor_cuda.run(face_vertices, mask, depth_map, batch_id, self.image_height, self.image_width)
        return octree_feature
