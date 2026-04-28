import os
import random
import numpy as np

def set_full_seed(seed: int, tensorflow=None):
    """
    Ustawia seedy możliwie "w pełni":
    - PYTHONHASHSEED
    - random
    - numpy
    - tensorflow 
    - wymusza deterministyczne operacje TF 
    """
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    # Flagi deterministycznosci TensorFlow ustawiane przed inicjalizacja GPU/TF.
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

    random.seed(seed)
    np.random.seed(seed)

    if tensorflow is not None:
        tensorflow.random.set_seed(seed)

        # Dodatkowa deterministycznosc, jesli funkcja jest dostepna.
        try:
            tensorflow.config.experimental.enable_op_determinism(True)
        except Exception:
            pass
