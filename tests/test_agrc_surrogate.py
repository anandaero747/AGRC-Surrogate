"""
Tests for agrc-surrogate.

Run with:  pytest tests/ -v
Slow tests (model inference) are marked with @pytest.mark.slow.
To skip them: pytest tests/ -v -m "not slow"
"""

import pytest
import numpy as np
from pathlib import Path

SAMPLE_AIRFOIL = Path(__file__).parent.parent / "agrc_surrogate" / "sc1095_full1.dat"


# ---------------------------------------------------------------------------
# Module-scoped fixtures: load models once for the entire test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_cst():
    from agrc_surrogate.opt_api import cst_from_airfoil
    return cst_from_airfoil(airfoil=str(SAMPLE_AIRFOIL))


@pytest.fixture(scope="module")
def sample_c81(sample_cst):
    from agrc_surrogate.opt_api import c81_from_cst
    return c81_from_cst(sample_cst)


# ---------------------------------------------------------------------------
# 1. Airfoil geometry reading
# ---------------------------------------------------------------------------

class TestGeometryReading:

    def test_read_returns_four_arrays(self):
        from agrc_surrogate.predict_360_pt3_pt8 import read_airfoil_single_dat_autofix
        xU, yU, xL, yL = read_airfoil_single_dat_autofix(str(SAMPLE_AIRFOIL))
        for arr in (xU, yU, xL, yL):
            assert isinstance(arr, np.ndarray)
            assert len(arr) > 0

    def test_upper_surface_above_lower_at_midchord(self):
        from agrc_surrogate.predict_360_pt3_pt8 import (
            read_airfoil_single_dat_autofix,
            _interp_branch_y_at_x,
        )
        xU, yU, xL, yL = read_airfoil_single_dat_autofix(str(SAMPLE_AIRFOIL))
        y_upper = _interp_branch_y_at_x(xU, yU, xq=0.5)
        y_lower = _interp_branch_y_at_x(xL, yL, xq=0.5)
        assert y_upper > y_lower, "Upper surface should be above lower at mid-chord"

    def test_coordinates_normalized_to_unit_chord(self):
        from agrc_surrogate.predict_360_pt3_pt8 import read_airfoil_single_dat_autofix
        xU, yU, xL, yL = read_airfoil_single_dat_autofix(str(SAMPLE_AIRFOIL))
        assert np.max(xU) <= 1.0 + 1e-6
        assert np.max(xL) <= 1.0 + 1e-6
        assert np.min(xU) >= 0.0 - 1e-6
        assert np.min(xL) >= 0.0 - 1e-6

    def test_too_few_points_raises(self, tmp_path):
        from agrc_surrogate.predict_360_pt3_pt8 import read_airfoil_single_dat_autofix
        sparse = tmp_path / "sparse.dat"
        sparse.write_text("0.0 0.0\n0.5 0.1\n1.0 0.0\n")
        with pytest.raises(Exception):
            read_airfoil_single_dat_autofix(str(sparse))

    def test_non_numeric_file_raises(self, tmp_path):
        from agrc_surrogate.predict_360_pt3_pt8 import read_airfoil_single_dat_autofix
        bad = tmp_path / "bad.dat"
        bad.write_text("NACA 0012\nthis is not a coordinate file\n")
        with pytest.raises(Exception):
            read_airfoil_single_dat_autofix(str(bad))


# ---------------------------------------------------------------------------
# 2. CST parameterization
# ---------------------------------------------------------------------------

class TestCSTParameterization:

    def test_cst_shape_is_20(self):
        from agrc_surrogate.opt_api import cst_from_airfoil
        cst = cst_from_airfoil(airfoil=str(SAMPLE_AIRFOIL))
        assert cst.shape == (20,)

    def test_cst_values_are_finite(self):
        from agrc_surrogate.opt_api import cst_from_airfoil
        cst = cst_from_airfoil(airfoil=str(SAMPLE_AIRFOIL))
        assert np.all(np.isfinite(cst)), "All CST coefficients must be finite"

    def test_missing_args_raises_value_error(self):
        from agrc_surrogate.opt_api import cst_from_airfoil
        with pytest.raises(ValueError, match="Provide either"):
            cst_from_airfoil()

    def test_chebyshev_cst_evaluates_correctly(self):
        from agrc_surrogate.predict_360_pt3_pt8 import CST_chebyshev_TE
        x = np.linspace(0.01, 1.0, 50)
        # Zero coefficients → zero shape → y = 0
        y = CST_chebyshev_TE(x, *np.zeros(10))
        assert y.shape == x.shape
        assert np.allclose(y, 0.0)

    def test_chebyshev_cst_nonzero_coefficients(self):
        from agrc_surrogate.predict_360_pt3_pt8 import CST_chebyshev_TE
        x = np.linspace(0.01, 1.0, 50)
        coeffs = np.ones(10) * 0.1
        y = CST_chebyshev_TE(x, *coeffs)
        assert not np.allclose(y, 0.0), "Non-zero coefficients should produce non-zero shape"

    def test_chebyshev_cst_te_offset_applied(self):
        from agrc_surrogate.predict_360_pt3_pt8 import CST_chebyshev_TE
        x = np.linspace(0.01, 1.0, 50)
        coeffs_no_offset = np.zeros(10)
        coeffs_with_offset = np.zeros(10)
        coeffs_with_offset[-1] = 0.05  # te_offset = 0.05
        y_no = CST_chebyshev_TE(x, *coeffs_no_offset)
        y_te = CST_chebyshev_TE(x, *coeffs_with_offset)
        # At x=1, te_offset * x = 0.05
        assert y_te[-1] == pytest.approx(0.05, abs=1e-6)
        assert not np.allclose(y_no, y_te)


# ---------------------------------------------------------------------------
# 3. Full prediction pipeline (loads TF models — marked slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPredictionPipeline:

    def test_output_keys_present(self, sample_c81):
        for key in ("aoa", "mach", "Cl", "Cd", "Cm", "cst"):
            assert key in sample_c81, f"Missing key '{key}' in c81_from_cst output"

    def test_output_shapes(self, sample_c81):
        assert sample_c81["aoa"].shape == (361,)
        assert sample_c81["Cl"].shape == (361, 8)
        assert sample_c81["Cd"].shape == (361, 8)
        assert sample_c81["Cm"].shape == (361, 8)
        assert sample_c81["mach"].shape == (8,)

    def test_aoa_spans_full_360(self, sample_c81):
        assert sample_c81["aoa"][0] == pytest.approx(-180.0)
        assert sample_c81["aoa"][-1] == pytest.approx(180.0)
        assert len(sample_c81["aoa"]) == 361

    def test_mach_values_correct(self, sample_c81):
        expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        np.testing.assert_array_almost_equal(sample_c81["mach"], expected)

    def test_drag_values_finite(self, sample_c81):
        assert np.all(np.isfinite(sample_c81["Cd"])), "Cd must be finite at all AoA and Mach"

    def test_all_outputs_finite(self, sample_c81):
        for key in ("Cl", "Cd", "Cm"):
            assert np.all(np.isfinite(sample_c81[key])), f"{key} contains non-finite values"

    def test_invalid_cst_size_raises(self):
        from agrc_surrogate.opt_api import c81_from_cst
        with pytest.raises(ValueError):
            c81_from_cst(np.zeros(15))

    def test_write_output_creates_file(self, sample_cst, tmp_path):
        from agrc_surrogate.opt_api import c81_from_cst
        import os
        orig_dir = Path.cwd()
        os.chdir(tmp_path)
        try:
            result = c81_from_cst(sample_cst, write_files=True, out_prefix="TEST")
            assert (tmp_path / "TEST_all_mach.dat").exists()
        finally:
            os.chdir(orig_dir)


# ---------------------------------------------------------------------------
# 4. Optimization API utilities
# ---------------------------------------------------------------------------

class TestOptAPI:

    def test_interp_at_midpoint(self):
        from agrc_surrogate.opt_api import interp_at
        aoa = np.array([-10.0, 0.0, 10.0])
        y = np.array([0.0, 1.0, 2.0])
        assert interp_at(aoa, y, 5.0) == pytest.approx(1.5)

    def test_interp_at_exact_node(self):
        from agrc_surrogate.opt_api import interp_at
        aoa = np.linspace(-180, 180, 361)
        y = np.zeros(361)
        y[180] = 1.0  # spike at AoA = 0
        assert interp_at(aoa, y, 0.0) == pytest.approx(1.0)

    def test_interp_at_extrapolation_clamps(self):
        from agrc_surrogate.opt_api import interp_at
        aoa = np.array([0.0, 5.0, 10.0])
        y = np.array([1.0, 2.0, 3.0])
        assert interp_at(aoa, y, -5.0) == pytest.approx(1.0)  # clamp to left
        assert interp_at(aoa, y, 20.0) == pytest.approx(3.0)  # clamp to right

    @pytest.mark.slow
    def test_objective_ld_returns_finite_scalar(self, sample_c81):
        from agrc_surrogate.opt_api import objective_ld_at_aoa
        val = objective_ld_at_aoa(sample_c81, aoa_target=2.0, mach_target=0.3)
        assert np.isfinite(float(val))

    @pytest.mark.slow
    def test_objective_ld_varies_with_aoa(self, sample_c81):
        from agrc_surrogate.opt_api import objective_ld_at_aoa
        val_2deg = objective_ld_at_aoa(sample_c81, aoa_target=2.0, mach_target=0.5)
        val_15deg = objective_ld_at_aoa(sample_c81, aoa_target=15.0, mach_target=0.5)
        assert val_2deg != pytest.approx(val_15deg), "Objective should differ across AoA"
