import time,contextlib

class Profile(contextlib.ContextDecorator):
    # Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self):
        self.t_list = []

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t_list.append(self.dt)

    def time(self):      
        return time.time()
