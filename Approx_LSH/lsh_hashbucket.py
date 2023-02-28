import math
import random

import numpy as np
from numpy import linalg as LA
import numpy.random



class LSH:
    def __init__(self, func, n, K, L, m):
        self.func = func
        self.K = K
        self.L = L
        self.M = m
        self.n_nodes = n
        self.sample_size = 0
        self.count = 0
        self.hash_buckets = HashBuckets(K, L, self.n_nodes)
    def stats(self):
        avg_size = self.sample_size // max(self.count, 1)
        self.sample_size = 0
        self.count = 0
        print("Avg. Sample Size:", avg_size)

    def bucket_stats(self):
        avg_bucket_length = []
        for i in range(self.L):
            length = 0
            table = self.hash_buckets.tables[i]
            for key in table.keys():
                length += len(table[key])

            avg = round(length / len(table.keys()))
            avg_bucket_length.append(avg)

        return avg_bucket_length
    # P(x)
    # m is empirically 3
    def embed_weight_vector_ny(self, vector):
        '''weight embedding P(w) = [w;(1-|w|*|w|)^0.5]'''
        n = vector.shape[0]
        v = vector.reshape((n, 1))
        # v -= v.mean()  #to be deleted
        rnd_vec = v / (LA.norm(v))
        n = vector.shape[0]
        vector = vector.reshape((n, 1)) #*0.83
        embedded_arr = np.zeros((n + 1, 1))
        i, j = np.indices((n, 1))
        norm = LA.norm(vector)
        np.put_along_axis(embedded_arr, i, vector, axis=0)

        # for i in range(self.M):
        embedded_arr[n, :] = np.sqrt(1-pow(norm, 2))

        

        return embedded_arr
    
    
    def embed_weight_vector(self, vector):
        n = vector.shape[0]
        vector = vector.reshape((n, 1)) #*0.83
        embedded_arr = np.zeros((n + self.M, 1))
        i, j = np.indices((n, 1))
        norm = LA.norm(vector)
        np.put_along_axis(embedded_arr, i, vector, axis=0)

        for i in range(self.M):
            embedded_arr[n+i, :] = pow(norm, pow(2, i+1))


        return embedded_arr
    

    def embed_query_ny(self, query):
        '''Query embedding P(q) = [q; 0]
        Nyshabour'''
        n = query.shape[0]
        v = query.reshape((n, 1))
        # v -= v.mean()  #to be deleted
        rnd_vec = v / (LA.norm(v))
        vector = rnd_vec.reshape((n, 1))
        embedded_arr = np.zeros((n + 1, 1))
        i, j = np.indices((n, 1))
        np.put_along_axis(embedded_arr, i, vector, axis=0)

        # for i in range(self.M):
        embedded_arr[n, :] = 0

        return embedded_arr

    # Q(q)
    # m is empirically 3
    def embed_query(self, query):
        n = query.shape[0]
        v = query.reshape((n, 1))
        rnd_vec = v / (LA.norm(v))
        vector = rnd_vec.reshape((n, 1))
        embedded_arr = np.zeros((n + self.M, 1))
        i, j = np.indices((n, 1))
        np.put_along_axis(embedded_arr, i, vector, axis=0)

        for i in range(self.M):
            embedded_arr[n+i, :] = 1/2
    
        return embedded_arr

    def insert(self, item_id, item):
        embeded_item = self.embed_weight_vector(item)
        fp = self.func.hashSignature(embeded_item)
        self.hash_buckets.insert(fp, item_id)

    def query_remove(self, item, label):
        embeded_item = self.embed_weight_vector(item)
        fp = self.func.hashSignature(embeded_item)
        result = self.hash_buckets.query(np.squeeze(fp))
        if label in result:
            result.remove(label)
        self.sample_size += len(result)
        self.count += 1
        return list(result)

    def query(self, item):
        #item n by 1
        embeded_item = self.embed_query(item)
        fp = self.func.hashSignature(embeded_item)
        res = set()
        result = res.union(self.hash_buckets.query(fp))
        while len(result) < int(0.05 * self.n_nodes):
            result.update([random.randint(0,self.n_nodes-1)])
        self.sample_size += len(result)
        self.count += 1
        return result

    def query_multi(self, items, N):

        embed_queries = lambda x: [self.embed_query(x[i,:].transpose()) for i in range(N)]
        fp = lambda x, f: [self.func.hashSignature(y) for y in f(x)]
        res = fp(items, embed_queries)
        # embeded_items = self.embed_query(item)
        return list(self.hash_buckets.query_multi(res, N))


    def clear(self):
        self.hash_buckets.clear()



class HashBuckets:

    def __init__(self, K, L, n):
        self.tables = []
        self.K = K
        self.L = L
        self.n_nodes = n
        for i in range(L):
            table ={}
            self.tables.append(table)

    def insert(self, fp, item_id):

        for idx in range (self.L):
            self.add(fp[idx], idx, item_id)

    def add(self, key, id, item_id):
        table = self.tables[id]
        if key not in table.keys():
            self.tables[id][key] = [item_id]

        else:
            self.tables[id][key].append(item_id)

    def query(self, keys):

        result = set()
        for i in range(self.L):
            if len(result) < int(0.05 * self.n_nodes):
                self.retrieve(result, i, keys[i])
            else:
                return result

        return result

    def query_multi(self, keys, N):
        result = set()
        for j in range(N):
            for i in range (self.L):
                self.retrieve(result, i, keys[j][i])

        return result

    def retrieve(self, res, table_id, key):
        table = self.tables[table_id]

        if key in table.keys():
            if len(table[key]) > int(0.05 * self.n_nodes):
                sample = random.sample(table[key], int(0.05 * self.n_nodes))
                res.update(sample)
            else:
                res.update(table[key])




    def clear(self):
        for i in range(self.L):
            self.tables[i] = {}
