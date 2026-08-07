"""The console's coordinate conversion, which every map position depends on."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from ui.bridge.projection import (
    ProjectionError,
    _parse_proj,
    build_projection,
    lonlat_to_utm,
    utm_to_lonlat,
)

LYTTON_LOCATION = {
    "netOffset": "-585470.77,-5558508.58",
    "convBoundary": "0.00,0.00,25208.62,42349.99",
    "origBoundary": "-121.872930,49.815387,-121.364416,50.866942",
    "projParameter": "+proj=utm +zone=10 +ellps=WGS84 +datum=WGS84 +units=m +no_defs",
}

HALIFAX_LOCATION = {
    "netOffset": "-385362.12,-4932502.56",
    "convBoundary": "0.00,0.00,80020.14,46049.74",
    "origBoundary": "-64.459427,44.543468,-62.988433,45.319489",
    "projParameter": "+proj=utm +zone=20 +ellps=WGS84 +datum=WGS84 +units=m +no_defs",
}


def fake_net(location):
    offset = [float(v) for v in location["netOffset"].split(",")]
    return SimpleNamespace(
        _location=dict(location),
        getLocationOffset=lambda: offset,
        hasGeoProj=lambda: False,
        convertXY2LonLat=None,
    )


def test_parses_utm_zone_and_hemisphere():
    assert _parse_proj("+proj=utm +zone=20 +ellps=WGS84") == ("utm", 20, True)
    assert _parse_proj("+proj=utm +zone=33 +south +ellps=WGS84") == ("utm", 33, False)
    assert _parse_proj("!") == ("none", None, True)
    assert _parse_proj("+proj=lcc +lat_1=49")[0] == "unsupported"


def test_central_meridian_maps_to_zone_centre():
    # A point on the false easting sits exactly on the zone's central meridian.
    lon, lat = utm_to_lonlat(500000.0, 4_900_000.0, 20)
    assert lon == pytest.approx(-63.0, abs=1e-9)
    assert 44.0 < lat < 45.0


def test_places_the_first_halifax_ignition_where_it_happened():
    """Juneberry Lane, the 28 May 2023 ignition point in Westwood Hills.

    The expected position was taken from SUMO's own ``convertGeo`` on the bundled
    Halifax network, which this implementation matched to within a millimetre
    across the network boundary.
    """
    lon, lat = utm_to_lonlat(385362.12 + 44877.0, 4932502.56 + 20092.0, 20)
    assert lon == pytest.approx(-63.880889, abs=1e-5)
    assert lat == pytest.approx(44.723335, abs=1e-5)


def test_round_trips_through_a_forward_projection():
    """Inverting a point and projecting it back returns the original metre."""

    def forward(lon_deg: float, lat_deg: float, zone: int) -> tuple[float, float]:
        a, f = 6378137.0, 1 / 298.257223563
        e2 = f * (2 - f)
        ep2 = e2 / (1 - e2)
        k0 = 0.9996
        lon = math.radians(lon_deg)
        lat = math.radians(lat_deg)
        lon0 = math.radians((zone - 1) * 6 - 180 + 3)
        n = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
        t = math.tan(lat) ** 2
        c = ep2 * math.cos(lat) ** 2
        aa = math.cos(lat) * (lon - lon0)
        m = a * (
            (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat
            - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * lat)
            + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * lat)
            - (35 * e2**3 / 3072) * math.sin(6 * lat)
        )
        easting = k0 * n * (aa + (1 - t + c) * aa**3 / 6 + (5 - 18 * t + t**2 + 72 * c - 58 * ep2) * aa**5 / 120) + 500000.0
        northing = k0 * (
            m
            + n
            * math.tan(lat)
            * (
                aa**2 / 2
                + (5 - t + 9 * c + 4 * c**2) * aa**4 / 24
                + (61 - 58 * t + t**2 + 600 * c - 330 * ep2) * aa**6 / 720
            )
        )
        return easting, northing

    for easting, northing, zone in [
        (430239.12, 4952594.56, 20),
        (585470.77, 5558508.58, 10),
        (500000.0, 4000000.0, 20),
    ]:
        lon, lat = utm_to_lonlat(easting, northing, zone)
        back_e, back_n = forward(lon, lat, zone)
        assert back_e == pytest.approx(easting, abs=0.01)
        assert back_n == pytest.approx(northing, abs=0.01)


def test_net_offset_is_removed_before_inverting():
    convert, report = build_projection(fake_net(HALIFAX_LOCATION))
    assert report["source"] == "builtin_utm"
    # Simulation origin corresponds to the recorded UTM offset.
    lon, lat = convert(0.0, 0.0)
    expected = utm_to_lonlat(385362.12, 4932502.56, 20)
    assert lon == pytest.approx(expected[0], abs=1e-9)
    assert lat == pytest.approx(expected[1], abs=1e-9)


def test_verification_rejects_a_converter_that_disagrees_with_sumo():
    def wrong_verifier(x, y):
        del x, y
        return (0.0, 0.0)

    with pytest.raises(ProjectionError, match="disagrees with SUMO"):
        build_projection(fake_net(LYTTON_LOCATION), verify=wrong_verifier)


def test_verification_passes_when_the_two_agree():
    net = fake_net(LYTTON_LOCATION)
    convert, _ = build_projection(net)
    _, report = build_projection(net, verify=lambda x, y: convert(x, y))
    assert report["verified"] is True
    assert report["max_disagreement_m"] == pytest.approx(0.0, abs=1e-6)


def test_a_network_without_a_projection_is_refused():
    location = dict(LYTTON_LOCATION, projParameter="!")
    with pytest.raises(ProjectionError, match="no geographic projection"):
        build_projection(fake_net(location))


class TestForwardProjection:
    """Authoring needs longitude and latitude turned back into simulation coordinates."""

    @pytest.mark.parametrize("zone,northern,lon0,lat0", [
        (20, True, -63.90, 44.73),
        (10, True, -121.58, 50.23),
        (33, False, 18.40, -33.90),
    ])
    def test_round_trips_within_a_centimetre(self, zone, northern, lon0, lat0):
        for dlon in (-1.5, -0.5, 0.0, 0.5, 1.5):
            for dlat in (-0.4, 0.0, 0.4):
                lon, lat = lon0 + dlon, lat0 + dlat
                east, north = lonlat_to_utm(lon, lat, zone, northern)
                back_lon, back_lat = utm_to_lonlat(east, north, zone, northern)
                dy = (back_lat - lat) * 111132.0
                dx = (back_lon - lon) * 111320.0 * math.cos(math.radians(lat))
                assert math.hypot(dx, dy) < 0.05

    def test_central_meridian_maps_to_the_false_easting(self):
        east, _north = lonlat_to_utm(-63.0, 44.73, 20, True)
        assert east == pytest.approx(500000.0, abs=1e-6)

    def test_southern_hemisphere_uses_the_false_northing(self):
        _east, north = lonlat_to_utm(18.4, -33.9, 33, False)
        assert north > 6_000_000.0
