import os
import sys
import logging


def configure_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


configure_utf8_stdio()

"""
entry point projektu

cisza dla tensorflow
global seed
Punkt startowy uruchamiajacy menu eksperymentow z modulu train_compare.
"""

# Konfiguracja przed importem TensorFlow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Ograniczenie komunikatow informacyjnych i ostrzezen.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # Stabilniejsze logowanie TensorFlow.

# Wyciszenie stderr podczas importu TensorFlow.
class _NullWriter:
    def write(self, _): 
        pass
    def flush(self):
        pass
    def close(self):
        pass

_old_stderr = sys.stderr
sys.stderr = _NullWriter()

import tensorflow as tf

# Przywrocenie stderr.
sys.stderr = _old_stderr
configure_utf8_stdio()

# Wyciszenie loggera TensorFlow.
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from  seed_utils import set_full_seed
import train_compare


def main():
    global_seed = 123

    set_full_seed(global_seed, tensorflow=tf)
    try:
        tf.config.experimental.enable_op_determinism(True)
    except Exception:
        pass

    print("TensorFlow:", tf.__version__)
    print("GLOBAL SEED:", global_seed)

    train_compare.main()


if __name__ == "__main__":
    main()

