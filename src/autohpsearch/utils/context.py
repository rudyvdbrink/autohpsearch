# %% libraries

import os
import sys
from contextlib import contextmanager

# %% context management

@contextmanager
def hush():
    with open(os.devnull, 'w', encoding='utf-8') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
# %%
