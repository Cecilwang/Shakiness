# Author: Cecil Wang (cecilwang@126.com)

import sys

class ProgressBar(object):

    def __init__(self, per_dot, per_line):
        self.per_dot = per_dot
        self.per_line = per_line
        self.count = 0

    def __call__(self, fn):
        def wrapped(*args, **kwargs):
            self.count += 1
            if self.count % self.per_dot == 0:
                print('.', end='')
                sys.stdout.flush()
            if self.count % self.per_line == 0:
                print()
                sys.stdout.flush()
            return fn(*args, **kwargs)
        return wrapped
