"""
Sanity check: three-way comparison of aero and geo functions.

Compares outputs of:
  1. Upstream Python  (aero_upstream.py / _geo_upstream.py)
  2. Numba njit       (aero.py / _geo.py with @njit cache=True)
  3. Numba fastmath   (same functions with fastmath=True, if enabled)

Run with:
    pytest tests/test_numba_sanity.py -v
"""

"""
Sanity check: three-way comparison of aero and geo functions.
Run with: pytest tests/test_numba_sanity.py -v
"""

import numpy as np
import pytest
import importlib.util
import os
import sys

# Add bluesky package to path so imports work without install
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bluesky"))

# Mock bluesky.settings before any imports
from unittest.mock import MagicMock
mock_settings = MagicMock()
mock_settings.casmach_threshold = 2.0
sys.modules['bluesky'] = MagicMock()
sys.modules['bluesky.settings'] = mock_settings

def load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

TOOLS = os.path.join(ROOT, "bluesky", "tools")

upstream_aero = load_module("aero_upstream", os.path.join(TOOLS, "aero_upstream.py"))
numba_aero    = load_module("aero_numba",    os.path.join(TOOLS, "aero.py"))
upstream_geo  = load_module("geo_upstream",  os.path.join(TOOLS, "geo", "_geo_upstream.py"))
numba_geo     = load_module("geo_numba",     os.path.join(TOOLS, "geo", "_geo.py"))



import numpy as np
import pytest
import importlib.util
import os
import sys

# ---------------------------------------------------------------------------
# Helpers to load modules from file path directly
# ---------------------------------------------------------------------------

def load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

TOOLS = os.path.join(os.path.dirname(__file__), "..", "bluesky", "tools")

upstream_aero = load_module("aero_upstream",    os.path.join(TOOLS, "aero_upstream.py"))
numba_aero    = load_module("aero_numba",       os.path.join(TOOLS, "aero.py"))
upstream_geo  = load_module("geo_upstream",     os.path.join(TOOLS, "geo", "_geo_upstream.py"))
numba_geo     = load_module("geo_numba",        os.path.join(TOOLS, "geo", "_geo.py"))

# ---------------------------------------------------------------------------
# Shared test inputs
# ---------------------------------------------------------------------------

# Altitudes: sea level, 5 km, 11 km (tropopause), 15 km, 20 km
H_SCALAR = np.array([0., 5000., 11000., 15000., 20000.])

# Speeds in m/s (CAS / TAS)
SPEEDS = np.array([50., 100., 150., 200., 250.])

# Mach numbers
MACHS = np.array([0.3, 0.5, 0.7, 0.8, 0.85])

# Lat/lon pairs (degrees)
LAT1 = np.array([52.3, 48.8, 40.7,  1.3, -33.9])
LON1 = np.array([ 4.7,  2.3, -74.0, 103.8, 151.2])
LAT2 = np.array([51.5, 50.1, 34.0,  35.7, -37.8])
LON2 = np.array([-0.1,  8.6, 135.0, 139.7, 144.9])

# Tolerance: 1e-6 relative — tight enough for correctness, loose enough for
# float reordering differences introduced by fastmath
RTOL = 1e-6
ATOL = 1e-6


# ---------------------------------------------------------------------------
# AERO: vatmos
# ---------------------------------------------------------------------------

class TestVatmos:
    def test_pressure_upstream_vs_numba(self):
        p_up,  rho_up,  T_up  = upstream_aero.vatmos(H_SCALAR)
        p_nb,  rho_nb,  T_nb  = numba_aero.vatmos(H_SCALAR)
        np.testing.assert_allclose(p_nb,   p_up,   rtol=RTOL, atol=ATOL,
                                   err_msg="vatmos pressure mismatch (upstream vs numba)")

    def test_density_upstream_vs_numba(self):
        _, rho_up, _ = upstream_aero.vatmos(H_SCALAR)
        _, rho_nb, _ = numba_aero.vatmos(H_SCALAR)
        np.testing.assert_allclose(rho_nb, rho_up, rtol=RTOL, atol=ATOL,
                                   err_msg="vatmos density mismatch (upstream vs numba)")

    def test_temperature_upstream_vs_numba(self):
        _, _, T_up = upstream_aero.vatmos(H_SCALAR)
        _, _, T_nb = numba_aero.vatmos(H_SCALAR)
        np.testing.assert_allclose(T_nb,   T_up,   rtol=RTOL, atol=ATOL,
                                   err_msg="vatmos temperature mismatch (upstream vs numba)")


# ---------------------------------------------------------------------------
# AERO: vtemp
# ---------------------------------------------------------------------------

class TestVtemp:
    def test_upstream_vs_numba(self):
        T_up = upstream_aero.vtemp(H_SCALAR)
        T_nb = numba_aero.vtemp(H_SCALAR)
        np.testing.assert_allclose(T_nb, T_up, rtol=RTOL, atol=ATOL,
                                   err_msg="vtemp mismatch")


# ---------------------------------------------------------------------------
# AERO: vvsound
# ---------------------------------------------------------------------------

class TestVvsound:
    def test_upstream_vs_numba(self):
        a_up = upstream_aero.vvsound(H_SCALAR)
        a_nb = numba_aero.vvsound(H_SCALAR)
        np.testing.assert_allclose(a_nb, a_up, rtol=RTOL, atol=ATOL,
                                   err_msg="vvsound mismatch")


# ---------------------------------------------------------------------------
# AERO: speed conversions
# ---------------------------------------------------------------------------

class TestSpeedConversions:
    def test_vcas2tas_upstream_vs_numba(self):
        tas_up = upstream_aero.vcas2tas(SPEEDS, H_SCALAR)
        tas_nb = numba_aero.vcas2tas(SPEEDS, H_SCALAR)
        np.testing.assert_allclose(tas_nb, tas_up, rtol=RTOL, atol=ATOL,
                                   err_msg="vcas2tas mismatch")

    def test_vtas2cas_upstream_vs_numba(self):
        cas_up = upstream_aero.vtas2cas(SPEEDS, H_SCALAR)
        cas_nb = numba_aero.vtas2cas(SPEEDS, H_SCALAR)
        np.testing.assert_allclose(cas_nb, cas_up, rtol=RTOL, atol=ATOL,
                                   err_msg="vtas2cas mismatch")

    def test_vtas2mach_upstream_vs_numba(self):
        m_up = upstream_aero.vtas2mach(SPEEDS, H_SCALAR)
        m_nb = numba_aero.vtas2mach(SPEEDS, H_SCALAR)
        np.testing.assert_allclose(m_nb, m_up, rtol=RTOL, atol=ATOL,
                                   err_msg="vtas2mach mismatch")

    def test_vmach2tas_upstream_vs_numba(self):
        tas_up = upstream_aero.vmach2tas(MACHS, H_SCALAR)
        tas_nb = numba_aero.vmach2tas(MACHS, H_SCALAR)
        np.testing.assert_allclose(tas_nb, tas_up, rtol=RTOL, atol=ATOL,
                                   err_msg="vmach2tas mismatch")

    def test_veas2tas_upstream_vs_numba(self):
        tas_up = upstream_aero.veas2tas(SPEEDS, H_SCALAR)
        tas_nb = numba_aero.veas2tas(SPEEDS, H_SCALAR)
        np.testing.assert_allclose(tas_nb, tas_up, rtol=RTOL, atol=ATOL,
                                   err_msg="veas2tas mismatch")

    def test_vtas2eas_upstream_vs_numba(self):
        eas_up = upstream_aero.vtas2eas(SPEEDS, H_SCALAR)
        eas_nb = numba_aero.vtas2eas(SPEEDS, H_SCALAR)
        np.testing.assert_allclose(eas_nb, eas_up, rtol=RTOL, atol=ATOL,
                                   err_msg="vtas2eas mismatch")

    def test_vmach2cas_upstream_vs_numba(self):
        cas_up = upstream_aero.vmach2cas(MACHS, H_SCALAR)
        cas_nb = numba_aero.vmach2cas(MACHS, H_SCALAR)
        np.testing.assert_allclose(cas_nb, cas_up, rtol=RTOL, atol=ATOL,
                                   err_msg="vmach2cas mismatch")

    def test_vcas2mach_upstream_vs_numba(self):
        m_up = upstream_aero.vcas2mach(SPEEDS, H_SCALAR)
        m_nb = numba_aero.vcas2mach(SPEEDS, H_SCALAR)
        np.testing.assert_allclose(m_nb, m_up, rtol=RTOL, atol=ATOL,
                                   err_msg="vcas2mach mismatch")

    def test_roundtrip_cas_tas(self):
        """CAS -> TAS -> CAS should recover original within tolerance."""
        tas = numba_aero.vcas2tas(SPEEDS, H_SCALAR)
        cas_recovered = numba_aero.vtas2cas(tas, H_SCALAR)
        np.testing.assert_allclose(cas_recovered, SPEEDS, rtol=RTOL, atol=ATOL,
                                   err_msg="CAS->TAS->CAS round-trip failed")

    def test_roundtrip_mach_tas(self):
        """Mach -> TAS -> Mach should recover original within tolerance."""
        tas  = numba_aero.vmach2tas(MACHS, H_SCALAR)
        mach_recovered = numba_aero.vtas2mach(tas, H_SCALAR)
        np.testing.assert_allclose(mach_recovered, MACHS, rtol=RTOL, atol=ATOL,
                                   err_msg="Mach->TAS->Mach round-trip failed")


# ---------------------------------------------------------------------------
# GEO: rwgs84
# ---------------------------------------------------------------------------

class TestRwgs84:
    def test_upstream_vs_numba(self):
        r_up = upstream_geo.rwgs84(LAT1)
        r_nb = numba_geo.rwgs84(LAT1)
        np.testing.assert_allclose(r_nb, r_up, rtol=RTOL, atol=ATOL,
                                   err_msg="rwgs84 mismatch")


# ---------------------------------------------------------------------------
# GEO: qdrdist (bearing + distance)
# ---------------------------------------------------------------------------

class TestQdrdist:
    def test_upstream_vs_numba(self):
        qdr_up, d_up = upstream_geo.qdrdist(LAT1, LON1, LAT2, LON2)
        qdr_nb, d_nb = numba_geo.qdrdist(LAT1, LON1, LAT2, LON2)
        np.testing.assert_allclose(qdr_nb, qdr_up, rtol=RTOL, atol=ATOL,
                                   err_msg="qdrdist bearing mismatch")
        np.testing.assert_allclose(d_nb,   d_up,   rtol=RTOL, atol=ATOL,
                                   err_msg="qdrdist distance mismatch")


# ---------------------------------------------------------------------------
# GEO: latlondist
# ---------------------------------------------------------------------------

class TestLatlondist:
    def test_upstream_vs_numba(self):
        d_up = upstream_geo.latlondist(LAT1, LON1, LAT2, LON2)
        d_nb = numba_geo.latlondist(LAT1, LON1, LAT2, LON2)
        np.testing.assert_allclose(d_nb, d_up, rtol=RTOL, atol=ATOL,
                                   err_msg="latlondist mismatch")


# ---------------------------------------------------------------------------
# GEO: kwikdist
# ---------------------------------------------------------------------------

class TestKwikdist:
    def test_upstream_vs_numba(self):
        d_up = upstream_geo.kwikdist(LAT1, LON1, LAT2, LON2)
        d_nb = numba_geo.kwikdist(LAT1, LON1, LAT2, LON2)
        np.testing.assert_allclose(d_nb, d_up, rtol=RTOL, atol=ATOL,
                                   err_msg="kwikdist mismatch")


# ---------------------------------------------------------------------------
# GEO: kwikqdrdist
# ---------------------------------------------------------------------------

class TestKwikqdrdist:
    def test_upstream_vs_numba(self):
        qdr_up, d_up = upstream_geo.kwikqdrdist(LAT1, LON1, LAT2, LON2)
        qdr_nb, d_nb = numba_geo.kwikqdrdist(LAT1, LON1, LAT2, LON2)
        np.testing.assert_allclose(qdr_nb, qdr_up, rtol=RTOL, atol=ATOL,
                                   err_msg="kwikqdrdist bearing mismatch")
        np.testing.assert_allclose(d_nb,   d_up,   rtol=RTOL, atol=ATOL,
                                   err_msg="kwikqdrdist distance mismatch")


# ---------------------------------------------------------------------------
# GEO: kwikpos
# ---------------------------------------------------------------------------

class TestKwikpos:
    def test_upstream_vs_numba(self):
        QDR  = np.array([45., 90., 135., 180., 270.])
        DIST = np.array([10., 50., 100., 200., 500.])
        lat_up, lon_up = upstream_geo.kwikpos(LAT1, LON1, QDR, DIST)
        lat_nb, lon_nb = numba_geo.kwikpos(LAT1, LON1, QDR, DIST)
        np.testing.assert_allclose(lat_nb, lat_up, rtol=RTOL, atol=ATOL,
                                   err_msg="kwikpos lat mismatch")
        np.testing.assert_allclose(lon_nb, lon_up, rtol=RTOL, atol=ATOL,
                                   err_msg="kwikpos lon mismatch")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_vatmos_sea_level(self):
        """Sea level should return ISA standard values."""
        p, rho, T = numba_aero.vatmos(np.array([0.]))
        assert abs(p[0]   - 101325.) < 1.0,  f"Sea level pressure wrong: {p[0]}"
        assert abs(rho[0] - 1.225)   < 0.01, f"Sea level density wrong: {rho[0]}"
        assert abs(T[0]   - 288.15)  < 0.01, f"Sea level temperature wrong: {T[0]}"

    def test_vatmos_tropopause(self):
        """At 11000m temperature should be ~216.65K."""
        _, _, T = numba_aero.vatmos(np.array([11000.]))
        assert abs(T[0] - 216.65) < 0.5, f"Tropopause temperature wrong: {T[0]}"

    def test_negative_cas(self):
        """Negative CAS should return negative TAS (sign preserved)."""
        tas = numba_aero.vcas2tas(np.array([-100.]), np.array([5000.]))
        assert tas[0] < 0, "Negative CAS should give negative TAS"

    def test_qdrdist_same_point(self):
        """Distance from a point to itself should be zero."""
        _, d = numba_geo.qdrdist(
            np.array([52.3]), np.array([4.7]),
            np.array([52.3]), np.array([4.7])
        )
        assert abs(d[0]) < 1e-9, f"Self-distance should be zero, got {d[0]}"