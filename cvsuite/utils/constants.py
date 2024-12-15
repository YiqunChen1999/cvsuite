
from importlib import util as imutil

if imutil.find_spec("torch") is not None:
    PYTORCH_AVAILABLE = True
else:
    PYTORCH_AVAILABLE = False

if imutil.find_spec("accelerate") is not None:
    ACCELERATE_AVAILABLE = True
else:
    ACCELERATE_AVAILABLE = False

if imutil.find_spec("detectron2") is not None:
    DETECTRON2_AVAILABLE = True
else:
    DETECTRON2_AVAILABLE = False

if imutil.find_spec("detrex") is not None:
    DETREX_AVAILABLE = True
else:
    DETREX_AVAILABLE = False
