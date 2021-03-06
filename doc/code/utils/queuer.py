# coding: utf-8

"""
The Queue function mainly deals with reading and preparing dataset in a multi-processing manner.
We didnot use the built-in tensorflow function Dataset because it lacks of flexibility.
The function defined below is mainly inspired by the Keras.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import sys
import threading
import numpy as np
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from contextlib import closing

try:
    import queue
except ImportError:
    import Queue as queue

# Global variables to be shared across processes.
_SHARED_GENERATORS = {}
# Value counter to provide unique id to different processes.
_GENERATOR_COUNTER = None


def init_pool_generator(gens, random_seed=None):
    global _SHARED_GENERATORS
    _SHARED_GENERATORS = gens

    if random_seed is not None:
        ident = mp.current_process().ident
        np.random.seed(random_seed + ident)


def next_sample(uid):
    return six.next(_SHARED_GENERATORS[uid])


class EnQueuer(object):
    def __init__(self, generator,
                 multiprocessing=True,  # whether use multi-processing, default True
                 random_seed=1234,      # control the randomness
                 ):
        """
        The queue that stores data using multiprocessing or multithreading.
        :param generator: A iterable batched-data generator, define the __next__ function
        """
        self.generator = generator
        self.multiprocessing = multiprocessing
        self.random_seed = random_seed

        global _GENERATOR_COUNTER
        if _GENERATOR_COUNTER is None:
            try:
                _GENERATOR_COUNTER = mp.Value("i", 0)
            except OSError:
                # OS system does not support multiprocessing
                _GENERATOR_COUNTER = 0

        if isinstance(_GENERATOR_COUNTER, int):
            self.uid = _GENERATOR_COUNTER
            _GENERATOR_COUNTER += 1
        else:

            with _GENERATOR_COUNTER.get_lock():
                self.uid = _GENERATOR_COUNTER.value
                _GENERATOR_COUNTER.value += 1

        self.workers = 0
        self.executor_fn = None
        self.queue = None
        self.run_thread = None
        self.stop_signal = None

    def is_running(self):
        # whether the queue is working
        return self.stop_signal is not None and not self.stop_signal.is_set()

    def _send_generator(self):
        # Send current Iterable to all workers
        _SHARED_GENERATORS[self.uid] = self.generator

    def _get_executor_init(self, workers):
        # Get the pool initializer for multiprocessing
        return lambda seqs: mp.Pool(workers,
                                    initializer=init_pool_generator,
                                    initargs=(seqs, self.random_seed))

    def start(self, workers=1, max_queue_size=10):
        """
        Start the handler's worker
        :param workers: number of worker threads
        :param max_queue_size: queue size,
            when full, workers could block on `put()`
        :return:
        """
        self.workers = workers if not self.multiprocessing else 1

        if self.multiprocessing:
            self.executor_fn = self._get_executor_init(self.workers)
        else:
            self.executor_fn = lambda _: ThreadPool(self.workers)

        self.queue = queue.Queue(max_queue_size)
        self.stop_signal = threading.Event()
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.daemon = True
        self.run_thread.start()

    def _run(self):
        # Submit request to the executor and queue the future data
        self._send_generator()  # share the initial generator
        with closing(self.executor_fn(_SHARED_GENERATORS)) as executor:
            while True:
                if self.stop_signal.is_set():
                    return
                self.queue.put(executor.apply_async(next_sample, (self.uid,)), block=True)

    def stop(self, timeout=None):
        # stop the running threads and wait for them to exist
        # should be called by the same thread which called `start()`
        self.stop_signal.set()
        with self.queue.mutex:
            self.queue.queue.clear()
            self.queue.unfinished_tasks = 0
            self.queue.not_full.notify()
        self.run_thread.join(timeout)
        _SHARED_GENERATORS[self.uid] = None

    def get(self):
        # Data fetcher
        try:
            while self.is_running():
                inputs = self.queue.get(block=True).get()
                self.queue.task_done()
                if inputs is not None:
                    yield inputs
        except StopIteration:
            # Special case for finite generators
            last_ones = []
            while self.queue.qsize() > 0:
                last_ones.append(self.queue.get(block=True))
            # wait for them to complete
            list(map(lambda f: f.wait(), last_ones))
            # keep the good ones
            last_ones = [future.get() for future in last_ones if future.successful()]
            for inputs in last_ones:
                if inputs is not None:
                    yield inputs
        except KeyboardInterrupt:
            self.stop()
            six.reraise(*sys.exc_info())
        except Exception as e:
            self.stop()
            if 'generator already executing' in str(e):
                raise RuntimeError('your generator is NOT thread-safe')
            six.reraise(*sys.exc_info())
