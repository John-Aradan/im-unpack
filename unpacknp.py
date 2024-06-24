import numpy as np

def unpack_column(A, B, scales, bit_width):
    scale = 2 ** (bit_width - 1)
    unpacked_A = [A.copy()]
    unpacked_B = [B.copy()]
    scales_list = [scales.copy()]

    while True:
        low_bit_vals = unpacked_A[-1] % scale
        high_bit_vals = unpacked_A[-1] // scale

        sparsity = np.mean((np.abs(unpacked_A[-1]) >= scale).astype(float), axis=0)
        sparsity_mask = sparsity > 0
        count_sparsity = np.sum(sparsity_mask).item()

        if count_sparsity == 0:
            break

        unpacked_A[-1][:, sparsity_mask] = low_bit_vals[:, sparsity_mask]
        unpacked_A.append(high_bit_vals[:, sparsity_mask])
        unpacked_B.append(unpacked_B[-1][:, sparsity_mask])
        scales_list.append(scales_list[-1][sparsity_mask] * scale)
    
    unpacked_A = np.concatenate(unpacked_A, axis=1)
    unpacked_B = np.concatenate(unpacked_B, axis=1)
    scales = np.concatenate(scales_list, axis=0)
    return unpacked_A, unpacked_B, scales

def unpack_row(A, bit_width):
    scale = 2 ** (bit_width - 1)
    unpacked_A = [A.copy()]
    indices = [np.arange(A.shape[0])]
    scales = [np.ones(A.shape[0], dtype=int)]

    while True:
        low_bit_vals = unpacked_A[-1] % scale
        high_bit_vals = unpacked_A[-1] // scale

        sparsity = np.mean((np.abs(unpacked_A[-1]) >= scale).astype(float), axis=1)
        sparsity_mask = sparsity > 0
        count_sparsity = np.sum(sparsity_mask).item()

        if count_sparsity == 0:
            break

        unpacked_A[-1][sparsity_mask, :] = low_bit_vals[sparsity_mask, :]
        unpacked_A.append(high_bit_vals[sparsity_mask, :])
        indices.append(indices[-1][sparsity_mask])
        scales.append(scales[-1][sparsity_mask] * scale)

    unpacked_A = np.concatenate(unpacked_A, axis=0)
    indices = np.concatenate(indices, axis=0)
    scales = np.concatenate(scales, axis=0)
    return unpacked_A, indices, scales

def expend_mat(M, size, dim):
    if dim == 0:
        extra = np.zeros((size, M.shape[1]), dtype=M.dtype)
        return np.concatenate([M, extra], axis=0)
    elif dim == 1:
        extra = np.zeros((M.shape[0], size), dtype=M.dtype)
        return np.concatenate([M, extra], axis=1)
    else:
        raise Exception()

def expend_vec(v, size):
    extra = np.zeros(size, dtype=v.dtype)
    return np.concatenate([v, extra], axis=0)

def unpack_both(A, B, scales, bit_width):
    scale = 2 ** (bit_width - 1)
    unpacked_A = A.copy()
    unpacked_B = B.copy()
    Pi_indices = np.arange(A.shape[0])
    Pi_scales = np.ones(A.shape[0], dtype=int)

    insert_pointer_i = A.shape[0]
    insert_pointer_j = A.shape[1]

    while True:
        sparsity_mask = np.abs(unpacked_A) >= scale
        col_sparsity = np.sum(sparsity_mask.astype(int), axis=1)
        row_sparsity = np.sum(sparsity_mask.astype(int), axis=0)

        col_val, col_idx = np.max(col_sparsity), np.argmax(col_sparsity)
        row_val, row_idx = np.max(row_sparsity), np.argmax(row_sparsity)

        if col_val == 0 and row_val == 0:
            break

        if col_val >= row_val:
            if insert_pointer_i >= unpacked_A.shape[0]:
                unpacked_A = expend_mat(unpacked_A, A.shape[0], 0)
                Pi_indices = expend_vec(Pi_indices, A.shape[0])
                Pi_scales = expend_vec(Pi_scales, A.shape[0])

            vals = unpacked_A[col_idx, :]

            unpacked_A[insert_pointer_i, :] = vals // scale
            unpacked_A[col_idx, :] = vals % scale

            Pi_indices[insert_pointer_i] = Pi_indices[col_idx]
            Pi_scales[insert_pointer_i] = Pi_scales[col_idx] * scale
            insert_pointer_i = insert_pointer_i + 1
        else:
            if insert_pointer_j >= unpacked_A.shape[1]:
                unpacked_A = expend_mat(unpacked_A, A.shape[1], 1)
                unpacked_B = expend_mat(unpacked_B, B.shape[1], 1)
                scales = expend_vec(scales, A.shape[1])

            vals = unpacked_A[:, row_idx]

            unpacked_A[:, insert_pointer_j] = vals // scale
            unpacked_A[:, row_idx] = vals % scale

            unpacked_B[:, insert_pointer_j] = unpacked_B[:, row_idx]
            scales[insert_pointer_j] = scales[row_idx] * scale
            insert_pointer_j = insert_pointer_j + 1

    unpacked_A = unpacked_A[:insert_pointer_i, :insert_pointer_j]
    unpacked_B = unpacked_B[:, :insert_pointer_j]
    Pi_indices = Pi_indices[:insert_pointer_i]
    Pi_scales = Pi_scales[:insert_pointer_i]
    scales = scales[:insert_pointer_j]
    return unpacked_A, unpacked_B, Pi_indices, Pi_scales, scales

def unpack(A, B, scales, bit_width, strategy):
    if strategy == unpack_row:
        A, Pi_indices, Pi_scales = unpack_row(A, bit_width)
    elif strategy == unpack_column:
        A, B, scales = unpack_column(A, B, scales, bit_width)
        Pi_indices = np.arange(A.shape[0])
        Pi_scales = np.ones(A.shape[0], dtype=A.dtype)
    elif strategy == unpack_both:
        A, B, Pi_indices, Pi_scales, scales = unpack_both(A, B, scales, bit_width)
    return A, B, Pi_indices, Pi_scales, scales

def scaled_matmul(unpacked_A, unpacked_B, scales):
    return np.dot(unpacked_A * scales[:, None], unpacked_B.T)

def pack_row(A, indices, scales):
    packed_A = np.zeros((indices.max() + 1, A.shape[1]), dtype=A.dtype)
    np.add.at(packed_A, indices, A * scales[:, None])
    return packed_A

def pack_transposed_row(A, indices, scales):
    packed_A = np.zeros((A.shape[0], indices.max() + 1), dtype=A.dtype)
    np.add.at(packed_A, (np.arange(A.shape[0])[:, None], indices), A * scales)
    return packed_A
