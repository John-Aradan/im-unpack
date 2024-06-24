from time import time
import jax
import jax.numpy as jnp
import math
import json
from unpackjx import unpack_row, unpack_column, unpack_both, unpack, scaled_matmul, pack_row, pack_transposed_row

st1 = time()

A = (jax.random.normal(jax.random.PRNGKey(0), (512, 1024)) * 8).astype(jnp.int32)
B = (jax.random.normal(jax.random.PRNGKey(1), (2048, 1024)) * 8).astype(jnp.int32)
C = jnp.dot(A, B.T)
bit_width = 4

scales = jnp.ones(A.shape[1], dtype=A.dtype)
Au, Be, APi_indices, APi_scales, scales_u = unpack(A, B, scales, bit_width, unpack_row)
Beu, Aue, BPi_indices, BPi_scales, scales_uu = unpack(Be, Au, scales_u, bit_width, unpack_row)

AueSuuBeu = scaled_matmul(Aue, Beu, scales_uu)
APiAueSuuBeu = pack_row(AueSuuBeu, APi_indices, APi_scales)
APiAueSuuBeuBPi = pack_transposed_row(APiAueSuuBeu, BPi_indices, BPi_scales)

print(jnp.max(jnp.abs(C - APiAueSuuBeuBPi)))

print(f"Finished JAX after: {(time() - st1)}")