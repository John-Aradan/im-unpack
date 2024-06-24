import jax
import jax.numpy as jnp

def unpack_column(A, B, scales, bit_width):
    scale = 2 ** (bit_width - 1)
    unpacked_A = [A.copy()]
    unpacked_B = [B.copy()]
    scales = [scales]
    
    while True:
        low_bit_vals = unpacked_A[-1] % scale
        high_bit_vals = unpacked_A[-1] // scale

        sparsity = jnp.mean((jnp.abs(unpacked_A[-1]) >= scale).astype(float), axis=0)
        sparsity_mask = sparsity > 0
        count_sparsity = jnp.sum(sparsity_mask).item()

        if count_sparsity == 0:
            break
        
        unpacked_A[-1] = unpacked_A[-1].at[:, sparsity_mask].set(low_bit_vals[:, sparsity_mask])
        unpacked_A.append(high_bit_vals[:, sparsity_mask])
        unpacked_B.append(unpacked_B[-1][:, sparsity_mask])
        scales.append(scales[-1][sparsity_mask] * scale)
    
    unpacked_A = jnp.concatenate(unpacked_A, axis=1)
    unpacked_B = jnp.concatenate(unpacked_B, axis=1)
    scales = jnp.concatenate(scales, axis=0)
    return unpacked_A, unpacked_B, scales

def unpack_row(A, bit_width):
    scale = 2 ** (bit_width - 1)
    unpacked_A = [A.copy()]
    indices = [jnp.arange(A.shape[0])]
    scales = [jnp.ones(A.shape[0], dtype=jnp.int32)]
    
    while True:
        low_bit_vals = unpacked_A[-1] % scale
        high_bit_vals = unpacked_A[-1] // scale
    
        sparsity = jnp.mean((jnp.abs(unpacked_A[-1]) >= scale).astype(float), axis=1)
        sparsity_mask = sparsity > 0
        count_sparsity = jnp.sum(sparsity_mask).item()

        if count_sparsity == 0:
            break
        
        unpacked_A[-1] = unpacked_A[-1].at[sparsity_mask, :].set(low_bit_vals[sparsity_mask, :])
        unpacked_A.append(high_bit_vals[sparsity_mask, :])
        indices.append(indices[-1][sparsity_mask])
        scales.append(scales[-1][sparsity_mask] * scale)

    unpacked_A = jnp.concatenate(unpacked_A, axis=0)
    indices = jnp.concatenate(indices, axis=0)
    scales = jnp.concatenate(scales, axis=0)
    return unpacked_A, indices, scales

def expend_mat(M, size, dim):
    if dim == 0:
        extra = jnp.zeros((size, M.shape[1]), dtype=M.dtype)
        return jnp.concatenate([M, extra], axis=0)
    elif dim == 1:
        extra = jnp.zeros((M.shape[0], size), dtype=M.dtype)
        return jnp.concatenate([M, extra], axis=1)
    else:
        raise Exception()

def expend_vec(v, size):
    extra = jnp.zeros(size, dtype=v.dtype)
    return jnp.concatenate([v, extra], axis=0)

def unpack_both(A, B, scales, bit_width):
    scale = 2 ** (bit_width - 1)
    unpacked_A = A.copy()
    unpacked_B = B.copy()
    Pi_indices = jnp.arange(A.shape[0])
    Pi_scales = jnp.ones(A.shape[0], dtype=jnp.int32)

    insert_pointer_i = A.shape[0]
    insert_pointer_j = A.shape[1]
    
    while True:
        sparsity_mask = jnp.abs(unpacked_A) >= scale
        col_sparsity = jnp.sum(sparsity_mask.astype(int), axis=1)
        row_sparsity = jnp.sum(sparsity_mask.astype(int), axis=0)
    
        col_val, col_idx = jnp.max(col_sparsity), jnp.argmax(col_sparsity)
        row_val, row_idx = jnp.max(row_sparsity), jnp.argmax(row_sparsity)

        if col_val == 0 and row_val == 0:
            break
    
        if col_val >= row_val:
            if insert_pointer_i >= unpacked_A.shape[0]:
                unpacked_A = expend_mat(unpacked_A, A.shape[0], 0)
                Pi_indices = expend_vec(Pi_indices, A.shape[0])
                Pi_scales = expend_vec(Pi_scales, A.shape[0])
                
            vals = unpacked_A[col_idx, :]
            
            unpacked_A = unpacked_A.at[insert_pointer_i, :].set(vals // scale)
            unpacked_A = unpacked_A.at[col_idx, :].set(vals % scale)
            
            Pi_indices = Pi_indices.at[insert_pointer_i].set(Pi_indices[col_idx])
            Pi_scales = Pi_scales.at[insert_pointer_i].set(Pi_scales[col_idx] * scale)
            insert_pointer_i += 1
        else:
            if insert_pointer_j >= unpacked_A.shape[1]:
                unpacked_A = expend_mat(unpacked_A, A.shape[1], 1)
                unpacked_B = expend_mat(unpacked_B, B.shape[1], 1)
                scales = expend_vec(scales, A.shape[1])
                
            vals = unpacked_A[:, row_idx]
            
            unpacked_A = unpacked_A.at[:, insert_pointer_j].set(vals // scale)
            unpacked_A = unpacked_A.at[:, row_idx].set(vals % scale)
            
            unpacked_B = unpacked_B.at[:, insert_pointer_j].set(unpacked_B[:, row_idx])
            scales = scales.at[insert_pointer_j].set(scales[row_idx] * scale)
            insert_pointer_j += 1

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
        Pi_indices = jnp.arange(A.shape[0])
        Pi_scales = jnp.ones(A.shape[0], dtype=jnp.int32)
    elif strategy == unpack_both:
        A, B, Pi_indices, Pi_scales, scales = unpack_both(A, B, scales, bit_width)
    return A, B, Pi_indices, Pi_scales, scales

def scaled_matmul(unpacked_A, unpacked_B, scales):
    return jnp.dot(unpacked_A.astype(float) * scales.astype(float)[:, None], unpacked_B.T.astype(float))

def pack_row(A, indices, scales):
    A, scales = A.astype(float), scales.astype(float)
    packed_A = jnp.zeros((indices.max() + 1, A.shape[1]), dtype=A.dtype)
    packed_A = packed_A.at[indices].add(A * scales[:, None])
    return packed_A

def pack_transposed_row(A, indices, scales):
    A, scales = A.astype(float), scales.astype(float)
    packed_A = jnp.zeros((A.shape[0], indices.max() + 1), dtype=A.dtype)
    packed_A = packed_A.at[:, indices].add(A * scales)
    return packed_A