"""Exit status only; kept as a file to avoid Windows shell quoting differences."""
import sys
try:
    import torch,nibabel,scipy,matplotlib
    sys.exit(0 if torch.cuda.is_available() else 1)
except ImportError:
    sys.exit(1)
