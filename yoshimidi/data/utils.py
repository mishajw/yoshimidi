import io
import sys
from contextlib import contextmanager


@contextmanager
def capture_output():
    stdout = sys.stdout
    stderr = sys.stderr
    in_memory = io.StringIO("")
    sys.stdout = in_memory
    sys.stderr = in_memory
    yield in_memory
    sys.stdout = stdout
    sys.stderr = stderr
