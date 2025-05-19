import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr

class TqdmStream:
    def write(self, msg):
        from tqdm import tqdm
        tqdm.write(msg, file=sys.__stdout__, end='')
    def flush(self):
        pass

@contextmanager
def tqdm_context():
    """
    Context manager that redirects stdout and stderr to tqdm.write,
    avoiding interference with progress bars.
    """
    stream = TqdmStream()
    with redirect_stdout(stream), redirect_stderr(stream):
        yield
