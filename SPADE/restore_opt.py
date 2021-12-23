"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""


# iter.txt <- 초기화


import time
import shutil
import os



from_iter_text='checkpoints/Flickr/opt(base)/iter.txt'
to_iter_text='checkpoints/Flickr/iter.txt'


shutil.copy(from_iter_text, to_iter_text)
