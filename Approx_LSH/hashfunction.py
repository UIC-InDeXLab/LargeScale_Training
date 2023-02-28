import numpy as np

class HashFunction:

    def __init__(self, L, b, d, m):
        # b = K
        # d = n_node_prev_layer
        self.m_L = L
        self.m = m
        self.randomMatrix = []
        self.hashes = np.empty(L, dtype=np.float)
        rand_gen = np.random.RandomState()
        for jdx in range(self.m_L):
            matrix = rand_gen.randn(d+self.m, b)
            positive = np.greater_equal(matrix, 0.0)
            negative = np.less(matrix, 0.0)
            result = positive.astype(np.float32) - negative.astype(np.float32)
            self.randomMatrix.append(result)




    def hashSignature(self, data):
        return RandomProjection(self.hashes, self.randomMatrix, data).run()



class RandomProjection:

    def __init__(self, hashes, projection_matrix, query):

        self.m_projection_matrix = projection_matrix
        self.m_query = query
        self.m_hashes = hashes

    def run(self):

        hash_idx = -1
        for projection in self.m_projection_matrix:
            dotProduct = np.matmul(projection.transpose(), self.m_query)
            signature = 0
            for idx in range(len(dotProduct)):
                signature |= self.sign(dotProduct[idx])
                signature <<= 1

            hash_idx += 1

            self.m_hashes[hash_idx] = signature

        return self.m_hashes


    def sign(self, value):

        if (value >= 0):
            return 1
        return 0
