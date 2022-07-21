from icecream import ic
import pickle
from utils.settings import settings
import os
import fnmatch
from tqdm import tqdm
from collections.abc import Sequence

class ChunkedList(Sequence):
    def __init__(self, lst=None, num_chunks=None, n=None, dirpath=None):
        self.is_big = bool(dirpath)
        
        if self.is_big:
            self.dirpath = dirpath
            self.chunks = sorted(fnmatch.filter(os.listdir(dirpath), '*.pkl'))
            num_chunks = len(self.chunks)
            self.k = n // num_chunks
            self.last_chunk_name, self.last_chunk = None, None
            self.n = n
        else:
            self.n = len(lst)
            self.k = self.n // (num_chunks if num_chunks > 0 else 1)
            self.chunks = [lst[i:i + self.k] for i in range(0, len(lst), self.k)]
        
    def get_chunk(self, i):
        if self.is_big:
            current_chunk_name = self.chunks[i // self.k]
            if self.last_chunk_name == current_chunk_name:
                current_chunk = self.last_chunk
            else:
                current_chunk = pickle.load(open(os.path.join(self.dirpath, current_chunk_name), 'rb'))
                self.last_chunk = current_chunk
                self.last_chunk_name  = current_chunk_name
            return current_chunk
        
        return self.chunks[i // self.k]
    
    def __getitem__(self, i):
        if isinstance(i, int):
            return self.get_chunk(i)[i % self.k]
        if isinstance(i, slice):
            return [self[k] for k in range(self.n)[i]]

    # def __iter__(self):
    #     self.iter_idx = 0
    #     return self

    # def __next__(self):
    #     if self.iter_idx < len(self):
    #         elt = self[self.iter_idx]
    #         self.iter_idx += 1
    #         return elt
    #     raise StopIteration

    def __len__(self):
        return self.n
    
    def apply(self, func, dirpath=None):
        # Only can only convert non big to big
        to_big = bool(dirpath)
        if to_big:
            for i in tqdm(list(range(len(self.chunks)))):
                chunk = self.get_chunk(i * self.k)
                if not os.path.exists(dirpath):
                    os.mkdir(dirpath)
                with open(os.path.join(dirpath, f'chunk{i}.pkl'), 'wb') as f:
                    pickle.dump(func(chunk), f)
            return ChunkedList(n=self.n, num_chunks=len(self.chunks), dirpath=dirpath)
        
        self.chunks = [func(chunk) for chunk in self.chunks]
        return self

    def to_big(self, dirpath):
        return self.apply(func=lambda x : x, dirpath=dirpath)

    def get_chunks(self):
        return [self.get_chunk(i * self.k) for i in range(len(self.chunks))]


# k = 100 # chunk size
# lst = range(10000)
# chunks = [lst[i:i + k] for i in range(0, len(lst), k)]
# dirpath = os.path.join(settings.cache_dir)


# cl = ChunkedList(lst, k)

# for i in cl:
#     print(i)
    
    
# scl = cl.apply(lambda l: [2*k for k in l], os.path.join(dirpath, 'scl'))

# for i in scl:
#     print(i)