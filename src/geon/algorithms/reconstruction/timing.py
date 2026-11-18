"""Monotonic reconstruction log timestamps, independent of Qt event delivery."""
import re
from time import perf_counter

_PREFIX=re.compile(r'^\[elapsed [0-9.]+s; delta [0-9.]+s\] ')


class ProgressTimer:
    def __init__(self, clock=perf_counter):
        self.clock=clock
        self.started=self.previous=clock()

    def format(self, message):
        now=self.clock()
        elapsed,delta=now-self.started,now-self.previous
        self.previous=now
        if _PREFIX.match(message):
            return message
        return f'[elapsed {elapsed:.3f}s; delta {delta:.3f}s] {message}'

    @staticmethod
    def message(entry):
        return _PREFIX.sub('',entry)
