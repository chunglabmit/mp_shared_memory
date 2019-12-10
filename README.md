# mp_shared_memory
A shared-memory wrapper backwards compatible with Python &lt; 3.8

[![Travis CI Status](https://travis-ci.org/chunglabmit/mp_shared_memory.svg?branch=master)](https://travis-ci.org/chunglabmit/mp_shared_memory)

This shared memory wrapper makes it efficient to share memory between
processes using the pickling mechanisms that underly the multiprocess pool
functional serialization / deserialization protocol. Python 3.8 has excellent
multiprocess shared memory support, but this package will work on Linux for
Python < 3.8 and its lifecycle model is a little more automatic to use than
multiprocess.shared_memory.

How to use:

```python
# A naive way to use multiprocessing to apply a filter across
# an image (of course you'd use padding if you did it for real)

import itertools
import multiprocessing
from scipy.ndimage import gaussian_filter
from mp_shared_memory import SharedMemory


def do_block(src, dest, x0, x1, y0, y1, sigma):
    dest[y0:y1, x0:x1] = gaussian_filter(src[y0:y1, x0:x1], sigma)


def mp_gaussian_filter(img, sigma):
    xr = range(0, img.shape[1], 64)
    yr = range(0, img.shape[0], 64)
    #
    # The scope of the shared memory is the function. The backing files
    # will be harvested soon after the function runs.
    #
    src = SharedMemory(img.shape, img.dtype)
    dest = SharedMemory(img.shape, img.dtype)
    with multiprocessing.Pool() as pool:
        futures = []
        for x0, y0 in itertools.product(xr, yr):
            x1 = min(x0+64, img.shape[1])
            y1 = min(y0+64, img.shape[0])
            futures.append(pool.apply_async(
                do_block, (src, dest, x0, x1, y0, y1, sigma)))
        for future in futures:
            future.get()
```
