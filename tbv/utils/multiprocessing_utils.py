"""
Copyright (c) 2021
Argo AI, LLC, All Rights Reserved.

Notice: All information contained herein is, and remains the property
of Argo AI. The intellectual and technical concepts contained herein
are proprietary to Argo AI, LLC and may be covered by U.S. and Foreign
Patents, patents in process, and are protected by trade secret or
copyright law. This work is licensed under a CC BY-NC-SA 4.0 
International License.

Originating Authors: John Lambert
"""


#!/usr/bin/python3

import math
import multiprocessing
from typing import Any, List


def send_list_to_workers_with_worker_id(num_processes: int, list_to_split: List[Any], worker_func_ptr, **kwargs):
    """Given a list of work, and a desired number of n workers, launch n worker processes
    that will each process 1/nth of the total work.

    Args:
        num_processes: integer, number of workers to launch
        list_to_split:
        worker_func_ptr: function pointer
        **kwargs:
    """
    jobs = []
    num_items = len(list_to_split)
    print(f"Will split {num_items} items between {num_processes} workers")
    chunk_sz = math.ceil(num_items / num_processes)
    for i in range(num_processes):

        start_idx = chunk_sz * i
        end_idx = start_idx + chunk_sz
        end_idx = min(end_idx, num_items)
        # print(f'{start_idx}->{end_idx}')

        worker_id = i
        p = multiprocessing.Process(target=worker_func_ptr, args=(list_to_split, start_idx, end_idx, worker_id, kwargs))
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()
