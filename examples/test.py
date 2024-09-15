import torch as th

from ofe import OctreeFeatureExtractor

if __name__ == '__main__':
    K = th.asarray([[572.41136339, 0., 325.2611084],
                         [0., 573.57043286, 242.04899588],
                         [0., 0., 1.]])
    ofe = OctreeFeatureExtractor(480, 640, 20.0, K)
    ofe.cuda()

    all_voxel_centers = th.ones(1, 20000, 3).cuda() * 2000
    all_voxel_centers[:, :, 0] = 0.0
    all_voxel_centers[:, :, 1] = 0.0
    depth_map = th.ones(1, 480, 640).cuda() * 100
    mask = th.ones(1, 480, 640).bool().cuda()

    octree_feature = ofe(all_voxel_centers, mask, depth_map)
    print(octree_feature)
