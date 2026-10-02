"""Execution-local host wall clocks; no GPU synchronization or global hooks."""
import time


class WallTimings:
    def __init__(self, clock=None):
        self.clock = clock or time.perf_counter
        self.events = []

    def call(self, phase, function, /, *args, **kwargs):
        start = self.clock()
        status = 'failed'
        try:
            value = function(*args, **kwargs)
            status = 'completed'
            return value
        finally:
            end = self.clock()
            self.events.append({'phase': phase, 'start': start, 'end': end,
                                'seconds': end - start, 'status': status})

    def report(self):
        return {'clock': 'perf_counter', 'events': [dict(e) for e in self.events],
                'scope': 'host call wall time; nested intervals overlap; not pure GPU kernel time',
                'gpu_synchronization_added': False,
                'historical_cache_time_counted_as_current': False}
