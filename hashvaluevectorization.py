import numpy as np

def HashValueVectorization(hash_values):
    #converting the hashvalues into binary vectors and return an array of bits
    binary_vectors = bin(int(hash_values, 16))[2:].zfill(len(hash_values) * 4)
    return np.array([int(bit) for bit in binary_vectors])

