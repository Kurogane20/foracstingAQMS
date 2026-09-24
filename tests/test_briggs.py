"""Koefisien dispersi harus sama persis dengan Briggs (1973) rural,
sebagaimana ditabelkan Hanna, Briggs & Hosker (1982)."""
import numpy as np
import pytest

from app.services.dispersion import _briggs_sigma_y, _briggs_sigma_z, _plume_conc

X = np.array([1000.0])

# Nilai rujukan dihitung tangan dari rumus tabel Briggs rural pada x = 1 km.
SIGMA_Z_1KM = {
    "A": 0.20 * 1000,                              # 200,0
    "B": 0.12 * 1000,                              # 120,0
    "C": 0.08 * 1000 * (1 + 0.2) ** -0.5,          #  73,0
    "D": 0.06 * 1000 * (1 + 1.5) ** -0.5,          #  37,9
    "E": 0.03 * 1000 * (1 + 0.3) ** -1,            #  23,1
    "F": 0.016 * 1000 * (1 + 0.3) ** -1,           #  12,3
}
SIGMA_Y_1KM = {c: a * 1000 * (1.1) ** -0.5
               for c, a in {"A": .22, "B": .16, "C": .11, "D": .08, "E": .06, "F": .04}.items()}


@pytest.mark.parametrize("stab", list("ABCDEF"))
def test_sigma_z_matches_briggs_rural(stab):
    assert _briggs_sigma_z(stab, X)[0] == pytest.approx(SIGMA_Z_1KM[stab], rel=1e-9)


@pytest.mark.parametrize("stab", list("ABCDEF"))
def test_sigma_y_matches_briggs_rural(stab):
    assert _briggs_sigma_y(stab, X)[0] == pytest.approx(SIGMA_Y_1KM[stab], rel=1e-9)


@pytest.mark.parametrize("stab", list("CDEF"))
def test_sigma_z_is_not_the_old_linear_form(stab):
    """Regresi: versi lama memakai az·x linear untuk semua kelas, sehingga
    konsentrasi jarak jauh pada kelas C–F diremehkan hingga ~2,8×."""
    x = np.array([5000.0])
    linear = {"C": 0.08, "D": 0.06, "E": 0.03, "F": 0.016}[stab] * 5000
    assert _briggs_sigma_z(stab, x)[0] < 0.75 * linear


def test_stable_night_plume_reaches_further_than_before():
    """Kelas F pada 3 km: dengan σz Briggs, konsentrasi relatif terhadap puncak
    dekat-sumber harus lebih tinggi daripada rumus linear lama."""
    x = np.array([100.0, 3000.0]); y = np.zeros(2)
    c = _plume_conc(x, y, Q=1.0, u=1.0, stab="F", sigma0_y=250.0, sigma0_z=40.0)
    ratio_new = c[1] / c[0]

    sz_lin = np.sqrt((0.016 * x) ** 2 + 40.0 ** 2)
    sy = np.sqrt(_briggs_sigma_y("F", x) ** 2 + 250.0 ** 2)
    c_lin = 1.0 / (sy * sz_lin)
    ratio_old = c_lin[1] / c_lin[0]

    assert ratio_new > 1.2 * ratio_old
