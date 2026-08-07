import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

import torch  # noqa: E402

torch.manual_seed(0)
