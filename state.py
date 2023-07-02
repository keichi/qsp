from typing import Sequence
from typing_extensions import TypeAlias

# import numpy as cp
import cupy as cp

from cupy.cuda.nvtx import RangePush, RangePop

StateVectorType: TypeAlias = "npt.NDArray[cp.cfloat]"


class QuantumState:
    def __init__(
        self,
        n_qubits: int,
        batch_size: int,
    ) -> None:
        self._dim = 2**n_qubits
        self._batch_size = batch_size
        self._vector = cp.zeros((batch_size, self._dim), dtype="complex128")
        self._vector[:, 0] = 1.0

    # @property
    # def vector(self) -> StateVectorType:
    #     return self._vector

    # TODO: just simple only single qubit and target only!
    def apply(self, targets: int, matrix):
        RangePush("apply")

        # only 1 qubit
        qubits = []
        qubits.append(targets)
        masks = self.mask_vec(qubits)
        qsize = 1

        indices = [self.indices_vec(i, qubits, masks) for i in range(self._dim >> qsize)]

        # indices.shape is (self._dim >> qsize, 2)
        indices = cp.asarray(indices)

        # values.shape is (self._bach_size, self._dim >> qsize, 2)
        values = self._vector[:, indices]

        # new_values.shape is (self._bach_size, self._dim >> qsize, 2)
        new_values = cp.einsum("kl,ijl->ijk", matrix, values)

        self._vector[:, indices] = new_values

        RangePop()

    def mask_vec(self, qubits: Sequence[int]):
        # only 1 qubit
        min_qubit_mask = 1 << qubits[0]
        max_qubit_mask = 1 << qubits[0]
        mask_low = min_qubit_mask - 1
        mask_high = ~(max_qubit_mask - 1)
        return [max_qubit_mask, mask_low, mask_high]

    def indices_vec(self, index: int, qubits: Sequence[int], masks: Sequence[int]):
        # only 1 qubit
        mask, mask_low, mask_high = masks[0], masks[1], masks[2]
        basis_0 = (index & mask_low) + ((index & mask_high) << len(qubits))
        basis_1 = basis_0 + mask
        return [basis_0, basis_1]
