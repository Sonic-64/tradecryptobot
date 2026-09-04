
import numpy as np
def rule_prediction(candle):
    signal_down = 0
    signal_up = 0
    if vol_z < 0:
        return 0
    if (abs(signal_up-signal_down))>2:
        return np.sign(signal_up-signal_down)