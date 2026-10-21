"""Small physical fixtures for the climatological water balance and river properties."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import unittest

import numpy as np

import terrain_hydrology_balance as hb


class WaterBalanceTests(unittest.TestCase):
    def test_arid_cells_have_zero_runoff_and_balance_closes_independently_of_chunks(self):
        precip = np.array([[80., 250., 900.], [1800., 3000., 400.]])
        temp = np.array([[27., 22., 14.], [25., 8., -2.]])
        lat = np.array([[22.], [-35.]])
        kwargs = dict(latitude_deg=lat, bio4=np.full((2, 3), 600.))
        result = hb.annual_water_balance(precip, temp, **kwargs)
        self.assertEqual(result["runoff_mm_year"].shape, (2, 3))
        self.assertEqual(result["runoff_mm_year"][0, 0], 0.0)  # no positive floor in a desert
        self.assertEqual(result["runoff_mm_year"][0, 1], 0.0)
        self.assertGreater(result["runoff_mm_year"][1, 1], 1500.0)
        self.assertTrue(result["converged"].all())
        closure = result["precipitation_mm_year"] - result["evapotranspiration_mm_year"] - result["runoff_mm_year"]
        np.testing.assert_allclose(closure, result["storage_change_mm"], atol=2e-3)
        self.assertLessEqual(np.abs(result["storage_change_mm"]).max(), hb.BalanceParams().tolerance_mm + 1e-4)
        np.testing.assert_allclose(result["precipitation_mm_year"], precip, rtol=1e-6)
        self.assertTrue((result["evapotranspiration_mm_year"] <= result["potential_evaporation_mm_year"] + 1e-3).all())
        single = hb.annual_water_balance(precip, temp, chunk_cells=1, **kwargs)
        for name in ("runoff_mm_year", "evapotranspiration_mm_year", "warmup_years"):
            np.testing.assert_array_equal(single[name], result[name])

    def test_orogen_half_years_are_exact_and_bio_inputs_reproduce_them_by_convention(self):
        ts, tw, ps, pw = 24., 2., 600., 180.
        orogen = hb.monthly_water_balance(None, None, latitude_deg=[40., -40.],
                                          temp_summer_c=[ts, tw], temp_winter_c=[tw, ts],
                                          precip_summer_mm=[ps, pw], precip_winter_mm=[pw, ps])
        rain, temp = orogen["precipitation_mm"], orogen["temperature_c"]
        np.testing.assert_allclose(rain[3:9].sum(0), [ps, pw])
        np.testing.assert_allclose(rain[[0, 1, 2, 9, 10, 11]].sum(0), [pw, ps])
        np.testing.assert_allclose(np.ptp(rain[3:9], axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(temp[6], [ts, tw])  # July = northern summer proxy
        np.testing.assert_allclose(temp[0], [tw, ts])
        # BIO definitions (dismo sample std, +1 mm) as in terrain_conditioning.orogen_bioclim.
        bio4 = temp.std(0, ddof=1) * 100
        bio15 = (rain + 1).std(0, ddof=1) / (rain + 1).mean(0) * 100
        np.testing.assert_allclose(bio4[0], abs(ts - tw) * 100 * np.sqrt(12 / 11) / (2 * np.sqrt(2)))
        np.testing.assert_allclose(bio15[0], abs(ps - pw) * 100 * np.sqrt(12 / 11) / (ps + pw + 12))
        bio = hb.monthly_water_balance([ps + pw] * 2, [(ts + tw) / 2] * 2, latitude_deg=[40., -40.],
                                       bio4=bio4, bio15=bio15, wet_half="summer")
        np.testing.assert_allclose(bio["temperature_c"], temp, atol=1e-9)
        np.testing.assert_allclose(bio["precipitation_mm"], rain, atol=1e-9)
        self.assertIn("convention", bio["metadata"]["precipitation_phase"])
        uniform = hb.monthly_water_balance(ps + pw, 12., latitude_deg=40., bio4=500.)
        np.testing.assert_allclose(uniform["precipitation_mm"], (ps + pw) / 12)
        self.assertTrue(uniform["metadata"]["precipitation_phase"].startswith("uniform"))

    def test_winter_snow_is_released_in_spring(self):
        common = dict(latitude_deg=55., precip_summer_mm=350., precip_winter_mm=350.)
        cold = hb.monthly_water_balance(None, None, temp_summer_c=16., temp_winter_c=-14., **common)
        mild = hb.monthly_water_balance(None, None, temp_summer_c=22., temp_winter_c=6., **common)
        snow = cold["snow_mm"]
        self.assertEqual(int(snow.argmax()), 2)  # pack peaks at end of March
        self.assertEqual(snow[6], 0.0)
        np.testing.assert_allclose(cold["melt_mm"].sum(), cold["snowfall_mm"].sum())
        self.assertIn(int(cold["runoff_mm"].argmax()), (3, 4))  # April/May melt pulse
        self.assertLess(cold["runoff_mm"][[0, 1, 11]].sum(), mild["runoff_mm"][[0, 1, 11]].sum())
        self.assertEqual(mild["snowfall_mm"].sum(), 0.0)

    def test_periodic_storage_and_perennial_ice_are_bounded(self):
        p = hb.BalanceParams()
        result = hb.monthly_water_balance([700., 300., 2500.], [9., -25., 20.], latitude_deg=[50., 80., -5.],
                                          bio4=[800., 400., 150.])
        self.assertTrue(result["converged"].all())
        self.assertLessEqual(result["warmup_years"].max(), p.max_years)
        self.assertLess(np.abs(result["residual_mm"]).max(), 1e-9)
        self.assertLessEqual(np.abs(result["storage_change_mm"].sum(0)).max(), p.tolerance_mm)
        # Ice never melts: the store sits at its cap and accumulation leaves as discharge proxy.
        np.testing.assert_allclose(result["snow_mm"][:, 1], p.snow_cap_mm)
        np.testing.assert_allclose(result["runoff_mm"][:, 1].sum(),
                                   300. - result["evapotranspiration_mm"][:, 1].sum(), rtol=1e-9)
        initial = result["initial_storage_mm"].sum(0)
        np.testing.assert_allclose(result["snow_mm"][-1] + result["soil_mm"][-1] + result["groundwater_mm"][-1],
                                   initial, atol=p.tolerance_mm)


class RoutingAndNavigationTests(unittest.TestCase):
    def test_monthly_routing_is_conservative_lossy_and_label_independent(self):
        downstream = np.array([2, 2, 3, -1, -1])
        lateral = np.array([[10., 0.], [5., 1.], [1., 0.], [2., 2.], [4., 0.]])
        loss = np.array([0., 0., 8., 0., 0.])
        routed = hb.route_monthly(downstream, lateral, loss_m3=loss)
        np.testing.assert_allclose(routed["outflow_m3"][2], [8., 0.])  # dry month: loss limited to 1 m3
        np.testing.assert_allclose(routed["loss_m3"][2], [8., 1.])
        np.testing.assert_allclose(routed["terminal_m3"], [14., 2.])
        np.testing.assert_allclose(routed["residual_m3"], 0, atol=1e-12)
        perm = np.array([3, 0, 4, 1, 2])  # new label -> old label
        inverse = np.argsort(perm)
        relabeled = np.where(downstream[perm] >= 0, inverse[np.maximum(downstream[perm], 0)], -1)
        again = hb.route_monthly(relabeled, lateral[perm], loss_m3=loss[perm])
        np.testing.assert_allclose(again["outflow_m3"], routed["outflow_m3"][perm])
        with self.assertRaises(ValueError):
            hb.route_monthly([1, 0], [1., 1.])
        stats = hb.flow_statistics(np.array([[0., 0., 5., 5., 5., 5., 5., 5., 5., 5., 0., 0.]]))
        self.assertEqual(stats["dry_months"][0], 4)
        self.assertEqual(stats["min_month_m3s"][0], 0.0)
        seasonal = hb.route_monthly([-1], [[100., 0.]], loss_m3=[[0., 50.]], conserve_annual_loss=True)
        np.testing.assert_allclose(seasonal['loss_m3'].sum(), 50.)
        np.testing.assert_allclose(seasonal['outflow_m3'], [[50., 0.]])
        np.testing.assert_allclose(seasonal['residual_m3'], 0, atol=1e-12)

    def test_navigation_requires_continuous_route_and_reports_bottleneck(self):
        # 0 -> 1 (falls) -> 2 -> 3 -> sea. Deep seasonal lowland river.
        downstream = np.array([1, 2, 3, -1])
        q = np.tile(np.where(np.arange(12) // 6 == 0, 3000., 200.), (4, 1))
        nav = hb.navigability(downstream, [5e3, 1e3, 4e4, 6e4], [1e-4] * 4, q,
                              outlet_navigable=[False, False, False, True],
                              obstacle=[False, True, False, False])
        canoe, boat, ship = (nav["vessels"].index(name) for name in ("canoe", "riverboat", "ship"))
        self.assertTrue(nav["navigable"][boat, 2:].all())
        np.testing.assert_allclose(nav["route_length_m"][boat, 2, 0], 1e5)
        self.assertFalse(nav["navigable"][:, :2].any())  # deep reach above the falls is cut off
        self.assertTrue((nav["limit_reach"][[canoe, boat], 0] == 1).all())
        self.assertTrue((nav["limit_reason"][[canoe, boat], 0] == hb.REASONS.index("obstacle")).all())
        self.assertTrue((nav["limit_reach"][ship, 0, 6:] == 0).all())  # first failure is its own depth
        self.assertTrue(nav["navigable"][ship, 3, :6].all())  # wet season only
        self.assertFalse(nav["navigable"][ship, 3, 6:].any())
        self.assertTrue((nav["limit_reason"][ship, 2, 6:] == hb.REASONS.index("depth")).all())
        self.assertEqual(nav["year_round_class"][3], boat)
        self.assertEqual(nav["seasonal_class"][3], ship)
        self.assertEqual(nav["year_round_class"][0], -1)
        closed = hb.navigability([-1], [1e4], [1e-5], q[:1], outlet_navigable=[False])
        self.assertTrue((closed["limit_reason"][:, 0] == hb.REASONS.index("no_outlet")).all())
        self.assertEqual(canoe, 0)
        # An inherited route may itself only be usable in some months.
        route_open = np.broadcast_to(np.arange(12) < 6, (1, 12))
        inherited = hb.navigability([-1], [1e4], [1e-5], q[:1], outlet_navigable=route_open)
        self.assertFalse(inherited['navigable'][:, 0, 6:].any())


if __name__ == "__main__":
    unittest.main()
