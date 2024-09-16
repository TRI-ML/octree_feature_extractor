#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>


__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old;
    old = !signbit(value) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
    return old;
}


__global__ void octree_feature_extractor_cuda_kernel(
        const float* faces,
        const bool* mask,
        const float* depth_map,
        const int* batch_id,
        const int num_faces,
        const int image_height,
        const int image_width,
        float* octree_feature) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_faces) {
        return;
    }
    const int ih = image_height;
    const int iw = image_width;
    const int bn = batch_id[i / 12];
    const int fn = i;

    const float* face = &faces[i * 9];

    /* pi[0], pi[1], pi[2] = leftmost, middle, rightmost points */
    int pi[3];
    if (face[0] < face[3]) {
        if (face[6] < face[0]) pi[0] = 2; else pi[0] = 0;
        if (face[3] < face[6]) pi[2] = 2; else pi[2] = 1;
    } else {
        if (face[6] < face[3]) pi[0] = 2; else pi[0] = 1;
        if (face[0] < face[6]) pi[2] = 2; else pi[2] = 0;
    }
    for (int k = 0; k < 3; k++) {
      if (pi[0] != k && pi[2] != k) {
          pi[1] = k;
      }
    }

    /* p[num][xyz]: x, y is normalized from [-1, 1] to [0, ih or iw - 1]. */
    float p[3][3];
    for (int num = 0; num < 3; num++) {
        for (int dim = 0; dim < 3; dim++) {
            if (dim == 0) {
                p[num][dim] = 0.5 * (face[3 * pi[num] + dim] * iw + iw - 1);
            } else if (dim == 1) {
                p[num][dim] = 0.5 * (face[3 * pi[num] + dim] * ih + ih - 1);
            } else {
                p[num][dim] = face[3 * pi[num] + dim];
            }
        }
    }
    if (p[0][0] == p[2][0]) return; // line, not triangle

    /* compute face_inv */
    float face_inv[9] = {
        p[1][1] - p[2][1], p[2][0] - p[1][0], p[1][0] * p[2][1] - p[2][0] * p[1][1],
        p[2][1] - p[0][1], p[0][0] - p[2][0], p[2][0] * p[0][1] - p[0][0] * p[2][1],
        p[0][1] - p[1][1], p[1][0] - p[0][0], p[0][0] * p[1][1] - p[1][0] * p[0][1]};

    float face_inv_denominator = (
        p[2][0] * (p[0][1] - p[1][1]) +
        p[0][0] * (p[1][1] - p[2][1]) +
        p[1][0] * (p[2][1] - p[0][1]));

    for (int k = 0; k < 9; k++) {
        face_inv[k] /= face_inv_denominator;
    }

    const int xi_min = max(ceil(p[0][0]), 0.);
    const int xi_max = min(p[2][0], iw - 1.0);

    for (int xi = xi_min; xi <= xi_max; xi++) {
        /* compute yi_min and yi_max */
        float yi1, yi2;
        if (xi <= p[1][0]) {
            if (p[1][0] - p[0][0] != 0) {
                yi1 = (p[1][1] - p[0][1]) / (p[1][0] - p[0][0]) * (xi - p[0][0]) + p[0][1];
            } else {
                yi1 = p[1][1];
            }
        } else {
            if (p[2][0] - p[1][0] != 0) {
                yi1 = (p[2][1] - p[1][1]) / (p[2][0] - p[1][0]) * (xi - p[1][0]) + p[1][1];
            } else {
                yi1 = p[1][1];
            }
        }
        yi2 = (p[2][1] - p[0][1]) / (p[2][0] - p[0][0]) * (xi - p[0][0]) + p[0][1];

        const int yi_min = max(0., ceil(min(yi1, yi2)));
        const int yi_max = min(max(yi1, yi2), ih - 1.0);

        for (int yi = yi_min; yi <= yi_max; yi++) {
            /* index in output buffers */
            int index = bn * ih * iw + yi * iw + xi;
            /* compute w = face_inv * p */
            float w[3];
            for (int k = 0; k < 3; k++) {
                w[k] = face_inv[3 * k + 0] * xi + face_inv[3 * k + 1] * yi + face_inv[3 * k + 2];
            }
            /* sum(w) -> 1, 0 < w < 1 */
            float w_sum = 0;
            for (int k = 0; k < 3; k++) {
                w[k] = min(max(w[k], 0.0), 1.0);
                w_sum += w[k];
            }
            for (int k = 0; k < 3; k++) w[k] /= w_sum;
            /* compute 1 / zp = sum(w / z) */
            const float zp = 1.0 / (w[0] / p[0][2] + w[1] / p[1][2] + w[2] / p[2][2]);
            const float zp_diff = zp - depth_map[index];
            const float float_mask = static_cast<float>(mask[index]);
            atomicMaxFloat(&octree_feature[(i / 12) * 2], zp_diff);
            atomicMaxFloat(&octree_feature[(i / 12) * 2 + 1], float_mask);
        }
    }
}


at::Tensor run_cuda(
        const at::Tensor& faces,
        const at::Tensor& mask,
        const at::Tensor& depth_map,
        const at::Tensor& batch_id,
        const int image_height,
        const int image_width) {

    const int num_faces = faces.size(0);
    const int threads = 512;

    auto float_opts = faces.options().dtype(at::kFloat);

    at::Tensor octree_feature = at::full({num_faces / 12, 2}, 0.0, float_opts);

    const dim3 blocks1 ((num_faces - 1) / threads +1);

    octree_feature_extractor_cuda_kernel<<<blocks1, threads>>>(
        faces.data_ptr<float>(),
        mask.data_ptr<bool>(),
        depth_map.data_ptr<float>(),
        batch_id.data_ptr<int>(),
        num_faces,
        image_height,
        image_width,
        octree_feature.data_ptr<float>());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)  {
        printf("Error in forward_face_index_map: %s\n", cudaGetErrorString(err));
    }

    return octree_feature;
}
