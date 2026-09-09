from contextlib import contextmanager
from time import perf_counter

@contextmanager
def timer():
    box = {}
    t0 = perf_counter()
    yield box
    box['seconds'] = perf_counter() - t0
