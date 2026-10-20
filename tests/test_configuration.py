"""Tests for rionmass.configuration: typed workflow inputs."""

from __future__ import annotations

import pytest

from rionmass.configuration import CalibrationConfig, ODSImportConfig, parse_isomer_energy_spec


class TestODSImportConfig:
    def test_defaults(self):
        cfg = ODSImportConfig(filename="x.ods", sheet_index=0)
        assert cfg.observable == "frequency"
        assert cfg.harmonic is None
        assert cfg.exclusion_list == ()
        assert cfg.unmeasured_list == ()

    def test_rejects_bad_observable(self):
        with pytest.raises(ValueError, match="observable must be"):
            ODSImportConfig(filename="x.ods", sheet_index=0, observable="bogus")

    def test_rejects_negative_sheet_index(self):
        with pytest.raises(ValueError, match="sheet_index"):
            ODSImportConfig(filename="x.ods", sheet_index=-1)

    def test_time_observable_accepted(self):
        cfg = ODSImportConfig(filename="x.ods", sheet_index=0, observable="time")
        assert cfg.observable == "time"


class TestCalibrationConfig:
    def test_defaults(self):
        cfg = CalibrationConfig(elbien_path="ElBiEn.dat")
        assert cfg.p == 3
        assert cfg.isomer_excitation_energy == {}
        assert cfg.elbien_source == "file"

    def test_rejects_p_below_one(self):
        with pytest.raises(ValueError, match="p .* must be >= 1"):
            CalibrationConfig(elbien_path="ElBiEn.dat", p=0)

    def test_rejects_iteration_max_below_one(self):
        with pytest.raises(ValueError, match="iterationMax must be >= 1"):
            CalibrationConfig(elbien_path="ElBiEn.dat", iterationMax=0)

    def test_file_source_requires_elbien_path(self):
        with pytest.raises(ValueError, match="elbien_path is required"):
            CalibrationConfig(elbien_source="file", elbien_path=None)

    def test_nist_source_does_not_require_elbien_path(self):
        cfg = CalibrationConfig(elbien_source="nist")
        assert cfg.elbien_path is None

    def test_rejects_unknown_source(self):
        with pytest.raises(ValueError, match="elbien_source must be"):
            CalibrationConfig(elbien_source="rodrigues")


class TestParseIsomerEnergySpec:
    def test_bare_energy(self):
        assert parse_isomer_energy_spec(["206mTl=2.6431"]) == {"206mTl": pytest.approx(2.6431)}

    def test_energy_with_uncertainty(self):
        result = parse_isomer_energy_spec(["205mAu=0.907,0.001"])
        assert result["205mAu"] == pytest.approx((0.907, 0.001))

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="Expected NAME=ENERGY_MEV"):
            parse_isomer_energy_spec(["206mTl2.6431"])

    def test_too_many_commas_raises(self):
        with pytest.raises(ValueError, match="Too many commas"):
            parse_isomer_energy_spec(["206mTl=1,2,3"])

    def test_multiple_entries(self):
        result = parse_isomer_energy_spec(["A=1.0", "B=2.0,0.1"])
        assert set(result) == {"A", "B"}
