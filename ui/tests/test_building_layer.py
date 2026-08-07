"""The building layer the Setup view draws and a selection box would hit-test against.

Selection is by centroid, matching the rule an operator drawing a box would expect, so
these tests pin centroid computation, the in-view filter, and how a package without any
building polygons degrades.
"""

from __future__ import annotations

import json

import pytest

from ui.tools.build_map_assets import (
    BUILDING_PAD_DEG,
    EdgeSnapper,
    LocalFrame,
    _attach_buildings,
    _buildings_file,
    _buildings_in_view,
    _snap_buildings,
)

# A square, an L shape, and a far-away square. Coordinates are longitude and latitude,
# which is what polyconvert writes under proj.plain-geo.
_POLYS = """<?xml version="1.0" encoding="UTF-8"?>
<additional>
  <poly id="square" shape="-63.90,44.70 -63.90,44.72 -63.88,44.72 -63.88,44.70"/>
  <poly id="ell" shape="-63.86,44.70 -63.86,44.74 -63.84,44.74 -63.84,44.72 -63.82,44.72 -63.82,44.70"/>
  <poly id="faraway" shape="-10.00,10.00 -10.00,10.02 -9.98,10.02 -9.98,10.00"/>
  <poly id="noshape"/>
</additional>
"""


def _polys(tmp_path):
    path = tmp_path / "buildings.xml"
    path.write_text(_POLYS)
    return path


class TestBuildingsInView:
    def test_centroid_is_the_vertex_average(self, tmp_path):
        rows = _buildings_in_view(_polys(tmp_path), [-64.0, 44.0, -63.0, 45.0])
        square = next(r for r in rows if r["id"] == "square")
        assert square["lon"] == -63.89
        assert square["lat"] == 44.71

    def test_filter_is_by_centroid_not_by_footprint(self, tmp_path):
        # A box covering only the left half of the square excludes it, because its
        # centroid at -63.89 sits outside, even though two of its corners do not.
        rows = _buildings_in_view(_polys(tmp_path), [-63.95, 44.60, -63.895, 44.80])
        assert [r["id"] for r in rows] == []

    def test_out_of_box_buildings_are_dropped(self, tmp_path):
        rows = _buildings_in_view(_polys(tmp_path), [-64.0, 44.0, -63.0, 45.0])
        assert "faraway" not in {r["id"] for r in rows}

    def test_shapeless_polygons_are_skipped(self, tmp_path):
        rows = _buildings_in_view(_polys(tmp_path), [-180.0, -90.0, 180.0, 90.0])
        assert "noshape" not in {r["id"] for r in rows}

    def test_footprint_is_carried(self, tmp_path):
        rows = _buildings_in_view(_polys(tmp_path), [-64.0, 44.0, -63.0, 45.0])
        ell = next(r for r in rows if r["id"] == "ell")
        assert len(ell["poly"]) == 6
        assert ell["poly"][0] == [-63.86, 44.7]

    def test_no_box_yields_nothing(self, tmp_path):
        assert _buildings_in_view(_polys(tmp_path), None) == []


class TestBuildingsFileResolution:
    def test_map_json_declaration_wins(self, tmp_path, monkeypatch):
        import ui.tools.build_map_assets as mod

        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        declared = tmp_path / "declared.xml"
        declared.write_text(_POLYS)
        cfg = {"map": {"buildings_file": "declared.xml", "sumo_cfg": "ignored.sumocfg"}}
        assert _buildings_file("pkg", cfg) == declared

    def test_declared_but_missing_returns_none(self, tmp_path, monkeypatch):
        import ui.tools.build_map_assets as mod

        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        cfg = {"map": {"buildings_file": "absent.xml"}}
        assert _buildings_file("pkg", cfg) is None

    def test_falls_back_to_sumocfg_additional_files(self, tmp_path, monkeypatch):
        import ui.tools.build_map_assets as mod

        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        (tmp_path / "sumo").mkdir()
        (tmp_path / "sumo" / "b.xml").write_text(_POLYS)
        (tmp_path / "sumo" / "x.sumocfg").write_text(
            '<sumoConfiguration><input><additional-files value="b.xml"/></input></sumoConfiguration>'
        )
        cfg = {"map": {"sumo_cfg": "sumo/x.sumocfg"}}
        assert _buildings_file("pkg", cfg) == (tmp_path / "sumo" / "b.xml").resolve()

    def test_no_sumo_cfg_returns_none(self, tmp_path, monkeypatch):
        import ui.tools.build_map_assets as mod

        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        assert _buildings_file("pkg", {"map": {}}) is None

    def test_missing_sumo_cfg_file_returns_none(self, tmp_path, monkeypatch):
        import ui.tools.build_map_assets as mod

        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        assert _buildings_file("pkg", {"map": {"sumo_cfg": "sumo/absent.sumocfg"}}) is None


class TestAttachBuildings:
    def _cfg(self, tmp_path, monkeypatch):
        import ui.tools.build_map_assets as mod

        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        (tmp_path / "b.xml").write_text(_POLYS)
        return {"map": {"buildings_file": "b.xml"}}

    def test_layer_is_padded_around_the_incident_bbox(self, tmp_path, monkeypatch):
        cfg = self._cfg(tmp_path, monkeypatch)
        preview = {"bbox": [-63.895, 44.705, -63.885, 44.715]}
        _attach_buildings(preview, "pkg", cfg)
        # The square's centroid is inside the padded box but outside the raw one.
        assert preview["buildings_meta"]["count"] == 1
        assert preview["buildings"][0]["id"] == "square"
        assert preview["buildings_meta"]["bbox"][0] < -63.895 + BUILDING_PAD_DEG

    def test_no_buildings_file_gives_an_empty_layer(self, tmp_path, monkeypatch):
        import ui.tools.build_map_assets as mod

        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        preview = {"bbox": [-64.0, 44.0, -63.0, 45.0]}
        _attach_buildings(preview, "pkg", {"map": {}})
        assert preview["buildings"] == []
        assert preview["buildings_meta"] == {"source": None, "count": 0, "footprints": False}

    def test_footprints_dropped_past_the_limit(self, tmp_path, monkeypatch):
        import ui.tools.build_map_assets as mod

        cfg = self._cfg(tmp_path, monkeypatch)
        monkeypatch.setattr(mod, "BUILDING_FOOTPRINT_LIMIT", 1)
        preview = {"bbox": [-63.90, 44.70, -63.82, 44.74]}
        _attach_buildings(preview, "pkg", cfg)
        assert preview["buildings_meta"]["count"] > 1
        assert preview["buildings_meta"]["footprints"] is False
        assert all("poly" not in b for b in preview["buildings"])
        assert all("lon" in b and "lat" in b for b in preview["buildings"])

    def test_layer_is_json_serialisable(self, tmp_path, monkeypatch):
        cfg = self._cfg(tmp_path, monkeypatch)
        preview = {"bbox": [-64.0, 44.0, -63.0, 45.0]}
        _attach_buildings(preview, "pkg", cfg)
        json.dumps(preview)  # must not raise


# --- Stage 0, nearest drivable edge per building ---

# Three horizontal segments at y = 0, 300, and 1000, plus one the snapper is told is
# undrivable. Distances from a point are therefore easy to state by hand.
_EDGES = {
    "near": [(0.0, 0.0), (1000.0, 0.0)],
    "mid": [(0.0, 300.0), (1000.0, 300.0)],
    "far": [(0.0, 1000.0), (1000.0, 1000.0)],
    "footpath": [(0.0, 10.0), (1000.0, 10.0)],
    "degenerate": [(5.0, 5.0)],
}
_DRIVABLE = {"near", "mid", "far"}


class TestEdgeSnapper:
    def _snapper(self):
        return EdgeSnapper(_EDGES, drivable=_DRIVABLE, cell_m=200.0)

    def test_picks_the_closest_edge(self):
        edge_id, dist = self._snapper().nearest(500.0, 20.0, 200.0)
        assert edge_id == "near"
        assert dist == pytest.approx(20.0)

    def test_picks_the_second_edge_when_nearer(self):
        edge_id, dist = self._snapper().nearest(500.0, 280.0, 200.0)
        assert edge_id == "mid"
        assert dist == pytest.approx(20.0)

    def test_beyond_max_distance_returns_nothing(self):
        assert self._snapper().nearest(500.0, 600.0, 200.0) == (None, None)

    def test_undrivable_edges_are_excluded(self):
        # The footpath at y=10 is nearer than "near" at y=0, and must be ignored.
        edge_id, dist = self._snapper().nearest(500.0, 9.0, 200.0)
        assert edge_id == "near"
        assert dist == pytest.approx(9.0)

    def test_single_point_edges_are_excluded(self):
        assert "degenerate" not in self._snapper().shapes

    def test_search_crosses_cell_boundaries(self):
        # A point far from its own cell's edges still finds one several rings away.
        edge_id, dist = self._snapper().nearest(500.0, 850.0, 200.0)
        assert edge_id == "far"
        assert dist == pytest.approx(150.0)

    def test_matches_brute_force_over_a_grid_of_queries(self):
        from sumolib import geomhelper

        snapper = self._snapper()
        for x in range(0, 1001, 137):
            for y in range(-100, 1101, 91):
                got_id, got_dist = snapper.nearest(float(x), float(y), 400.0)
                best_id, best = None, float("inf")
                for edge_id in _DRIVABLE:
                    _, d = geomhelper.polygonOffsetAndDistanceToPoint(
                        (float(x), float(y)), _EDGES[edge_id], perpendicular=False
                    )
                    if d < best:
                        best, best_id = float(d), edge_id
                if best > 400.0:
                    assert got_id is None
                else:
                    assert got_id == best_id, f"at ({x},{y})"
                    assert got_dist == pytest.approx(round(best, 1))

    def test_no_drivable_filter_uses_every_edge(self):
        snapper = EdgeSnapper(_EDGES, drivable=None, cell_m=200.0)
        edge_id, _ = snapper.nearest(500.0, 9.0, 200.0)
        assert edge_id == "footpath"


# Snapping happens in a local metric frame, because longitude and latitude cannot be
# turned into network XY without pyproj.  This projector is the inverse of that frame at
# a fixed reference point, so an edge stated in metres round-trips back to the same
# metres and a test can state the expected distance directly.
_REF_LON, _REF_LAT = -63.90, 44.73
_REF_FRAME = LocalFrame(_REF_LON, _REF_LAT)


class _FakeProjector:
    """Maps the metric test coordinates to longitude and latitude and back again."""

    @staticmethod
    def to_lonlat(x, y):
        return (
            _REF_LON + float(x) / _REF_FRAME.m_per_deg_lon,
            _REF_LAT + float(y) / _REF_FRAME.m_per_deg_lat,
        )

    @classmethod
    def line_to_lonlat(cls, shape):
        return [list(cls.to_lonlat(p[0], p[1])) for p in shape]


def _lonlat(x, y):
    return _FakeProjector.to_lonlat(x, y)


class _FakeLane:
    def __init__(self, ok):
        self.ok = ok

    def allows(self, _cls):
        return self.ok


class _FakeEdge:
    def __init__(self, edge_id, ok, road_type="highway.residential"):
        self.edge_id = edge_id
        self.ok = ok
        self.road_type = road_type

    def getID(self):
        return self.edge_id

    def getType(self):
        return self.road_type

    def getLanes(self):
        return [_FakeLane(self.ok)]


class _FakeNet:
    """Just enough network to answer which edges a household may spawn onto."""

    def __init__(self, drivable, types=None):
        self.drivable = set(drivable)
        self.types = types or {}

    def getEdges(self, withInternal=False):
        return [
            _FakeEdge(e, e in self.drivable, self.types.get(e, "highway.residential"))
            for e in _EDGES
        ]


class _View:
    def __init__(self, net, edge_shape):
        self.net = net
        self.EDGE_SHAPE = edge_shape


def _building(x, y, bid="b1"):
    lon, lat = _lonlat(x, y)
    return {"id": bid, "lon": lon, "lat": lat}


class TestSnapBuildings:
    def test_attaches_edge_and_distance(self):
        buildings = [_building(500.0, 20.0)]
        report = _snap_buildings(buildings, _View(None, _EDGES), _FakeProjector)
        # No network means no drivable filter, so the nearest edge of any kind wins.
        assert buildings[0]["edge"] == "footpath"
        assert buildings[0]["edge_dist_m"] == pytest.approx(10.0, abs=0.2)
        assert report == {"snapped": 1, "unsnapped": 0}

    def test_drivable_filter_is_applied(self):
        buildings = [_building(500.0, 9.0)]
        _snap_buildings(buildings, _View(_FakeNet(_DRIVABLE), _EDGES), _FakeProjector)
        assert buildings[0]["edge"] == "near"

    def test_limited_access_roads_are_never_chosen(self):
        # "near" is the closest road, but a driveway does not meet a motorway, so the
        # household is matched to the community street further out instead.
        # At y=120 the motorway is 120 m away and the community street 180 m, so the
        # nearer road loses to the one a driveway can actually meet.
        net = _FakeNet(_DRIVABLE, types={"near": "highway.motorway"})
        buildings = [_building(500.0, 120.0)]
        _snap_buildings(buildings, _View(net, _EDGES), _FakeProjector)
        assert buildings[0]["edge"] == "mid"
        assert buildings[0]["edge_dist_m"] == pytest.approx(180.0, abs=0.5)

    def test_a_building_with_only_a_motorway_in_range_is_left_unsnapped(self):
        net = _FakeNet({"near"}, types={"near": "highway.motorway"})
        buildings = [_building(500.0, 20.0)]
        report = _snap_buildings(buildings, _View(net, _EDGES), _FakeProjector)
        assert buildings[0]["edge"] is None
        assert report == {"snapped": 0, "unsnapped": 1}

    def test_out_of_range_building_gets_null_edge(self):
        buildings = [_building(500.0, 5000.0)]
        report = _snap_buildings(buildings, _View(None, _EDGES), _FakeProjector)
        assert buildings[0]["edge"] is None
        assert buildings[0]["edge_dist_m"] is None
        assert report == {"snapped": 0, "unsnapped": 1}

    def test_no_projector_leaves_every_building_unsnapped(self):
        buildings = [_building(500.0, 20.0)]
        report = _snap_buildings(buildings, _View(None, _EDGES), None)
        assert buildings[0]["edge"] is None
        assert report == {"snapped": 0, "unsnapped": 1}

    def test_no_view_leaves_every_building_unsnapped(self):
        buildings = [_building(500.0, 20.0)]
        report = _snap_buildings(buildings, None, _FakeProjector)
        assert buildings[0]["edge"] is None
        assert report == {"snapped": 0, "unsnapped": 1}

    def test_mixed_batch_reports_both_counts(self):
        buildings = [_building(500.0, 20.0, "in"), _building(500.0, 5000.0, "out")]
        report = _snap_buildings(buildings, _View(None, _EDGES), _FakeProjector)
        assert report == {"snapped": 1, "unsnapped": 1}


class TestLocalFrame:
    def test_origin_maps_to_zero(self):
        assert _REF_FRAME.to_m(_REF_LON, _REF_LAT) == pytest.approx((0.0, 0.0))

    def test_scales_are_plausible_at_mid_latitude(self):
        # A degree of latitude is about 111 km everywhere, a degree of longitude
        # shrinks with the cosine, so at 44.73 degrees it is around 79 km.
        assert 110_900 < _REF_FRAME.m_per_deg_lat < 111_400
        assert 78_500 < _REF_FRAME.m_per_deg_lon < 79_500

    def test_round_trip_preserves_distance(self):
        lon, lat = _lonlat(300.0, 400.0)
        x, y = _REF_FRAME.to_m(lon, lat)
        assert (x, y) == pytest.approx((300.0, 400.0), abs=0.01)

    def test_meta_reports_the_snap_counts(self, tmp_path, monkeypatch):
        import ui.tools.build_map_assets as mod

        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        (tmp_path / "b.xml").write_text(_POLYS)
        preview = {"bbox": [-64.0, 44.0, -63.0, 45.0]}
        _attach_buildings(preview, "pkg", {"map": {"buildings_file": "b.xml"}}, view=None)
        meta = preview["buildings_meta"]
        assert meta["snapped"] == 0
        assert meta["unsnapped"] == meta["count"]
        assert meta["snap_max_m"] == mod.BUILDING_SNAP_MAX_M
