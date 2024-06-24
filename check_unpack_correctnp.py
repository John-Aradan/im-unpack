import numpy as np
from unpacknp import unpack_row, unpack_column, unpack_both, unpack, scaled_matmul, pack_row, pack_transposed_row

st3 = time()

A = (np.random.randn(512, 1024) * 8).astype(int)
B = (np.random.randn(2048, 1024) * 8).astype(int)
C = np.dot(A, B.T)
bit_width = 4

scales = np.ones(A.shape[1], dtype=A.dtype)
Au, Be, APi_indices, APi_scales, scales_u = unpack(A, B, scales, bit_width, unpack_row)
Beu, Aue, BPi_indices, BPi_scales, scales_uu = unpack(Be, Au, scales_u, bit_width, unpack_row)

AueSuuBeu = scaled_matmul(Aue, Beu, scales_uu)
APiAueSuuBeu = pack_row(AueSuuBeu, APi_indices, APi_scales)
APiAueSuuBeuBPi = pack_transposed_row(APiAueSuuBeu, BPi_indices, BPi_scales)

print(np.max(np.abs(C - APiAueSuuBeuBPi)))

print(f"Finished Numpy after: {(time() - st3)}")
