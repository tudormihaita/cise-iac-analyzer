from __future__ import annotations

import numpy as np
import pytest

from tfimpact.evaluation import stats as S


def test_a12_hand_computed():
    # pairs: (3>1,3>2,3=3) (2>1,2=2,2<3) (1=1,1<2,1<3) -> 3 wins + 3 ties of 9
    assert S.a12([3, 2, 1], [1, 2, 3]) == pytest.approx((3 + 0.5 * 3) / 9)
    assert S.a12([5, 6], [1, 2]) == 1.0
    assert S.a12([1, 2], [5, 6]) == 0.0


def test_a12_magnitude_thresholds():
    assert S.a12_magnitude(0.52) == "negligible"
    assert S.a12_magnitude(0.60) == "small"
    assert S.a12_magnitude(0.30) == "medium"
    assert S.a12_magnitude(0.75) == "large"


def test_holm_known_example():
    # sorted p: 0.01, 0.02, 0.04 -> 3*0.01=0.03, 2*0.02=0.04, 1*0.04=0.04 (monotone)
    adj = S.holm([0.04, 0.01, 0.02])
    assert np.allclose(adj, [0.04, 0.03, 0.04])
    assert np.allclose(S.holm([0.5, 0.9]), [1.0, 1.0])


def test_wilcoxon_identical_and_clear_difference():
    x = np.linspace(0.1, 0.9, 20)
    assert S.wilcoxon_p(x, x) == 1.0
    assert S.wilcoxon_p(x + 0.05, x) < 0.001
    with pytest.raises(ValueError):
        S.wilcoxon_p([1, 2], [1])
