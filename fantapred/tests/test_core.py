import numpy as np
from fantapred.feature_engineering import _exp_weighted_mean, _apply_growth_decline_cap

def test_exp_weighted_mean():
    vals = [10, 20, 30]
    assert np.isclose(_exp_weighted_mean(vals, [0.6,0.3,0.1]), 10*0.1 + 20*0.3 + 30*0.6)

def test_growth_cap():
    assert _apply_growth_decline_cap(150, 100, "fmv") == 120  # 20% cap
