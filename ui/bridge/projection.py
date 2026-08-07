"""Simulation coordinates to longitude and latitude, with no new dependency.

Both bundled networks are projected with UTM on the WGS84 ellipsoid, which the
network file records in its ``projParameter`` string. ``sumolib`` can invert that
projection, but only when ``pyproj`` is installed, and the simulator itself never
needs it. So the inverse transverse Mercator series is implemented here directly.

The series is the standard one from USGS Professional Paper 1395 and is accurate
to a few millimetres inside a zone, far below the metre-scale precision a map of
an evacuation needs. Where ``pyproj`` happens to be available the sumolib route
is preferred, because it handles projections beyond UTM.

Whichever route is taken, :func:`build_projection` checks the result against
SUMO's own ``convertGeo`` on sample points before the console trusts it.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, List, Optional, Sequence, Tuple

# WGS84 ellipsoid and the UTM scale factor.
_A = 6378137.0
_F = 1.0 / 298.257223563
_K0 = 0.9996
_E2 = _F * (2.0 - _F)
_EP2 = _E2 / (1.0 - _E2)
_FALSE_EASTING = 500000.0
_FALSE_NORTHING = 10000000.0

#: Sample points may differ from SUMO's own conversion by at most this much, in
#: metres, before the console refuses to draw geography.
AGREEMENT_TOLERANCE_M = 5.0


class ProjectionError(Exception):
    """Raised when a network carries no usable geographic projection."""


def _parse_proj(proj_parameter: str) -> Tuple[str, Optional[int], bool]:
    """Return ``(kind, utm_zone, northern)`` for a PROJ parameter string."""
    text = (proj_parameter or "").strip()
    if not text or text == "!":
        return "none", None, True
    if "+proj=utm" in text:
        match = re.search(r"\+zone=(\d+)", text)
        zone = int(match.group(1)) if match else None
        northern = "+south" not in text
        return "utm", zone, northern
    if "+proj=longlat" in text or "+proj=latlong" in text:
        return "longlat", None, True
    return "unsupported", None, True


def utm_to_lonlat(easting: float, northing: float, zone: int, northern: bool = True) -> Tuple[float, float]:
    """Invert a UTM coordinate on the WGS84 ellipsoid."""
    x = easting - _FALSE_EASTING
    y = northing if northern else northing - _FALSE_NORTHING

    m = y / _K0
    mu = m / (_A * (1.0 - _E2 / 4.0 - 3.0 * _E2 ** 2 / 64.0 - 5.0 * _E2 ** 3 / 256.0))
    e1 = (1.0 - math.sqrt(1.0 - _E2)) / (1.0 + math.sqrt(1.0 - _E2))

    phi1 = (
        mu
        + (3.0 * e1 / 2.0 - 27.0 * e1 ** 3 / 32.0) * math.sin(2.0 * mu)
        + (21.0 * e1 ** 2 / 16.0 - 55.0 * e1 ** 4 / 32.0) * math.sin(4.0 * mu)
        + (151.0 * e1 ** 3 / 96.0) * math.sin(6.0 * mu)
        + (1097.0 * e1 ** 4 / 512.0) * math.sin(8.0 * mu)
    )

    sin_phi1 = math.sin(phi1)
    cos_phi1 = math.cos(phi1)
    tan_phi1 = math.tan(phi1)

    c1 = _EP2 * cos_phi1 ** 2
    t1 = tan_phi1 ** 2
    denom = math.sqrt(1.0 - _E2 * sin_phi1 ** 2)
    n1 = _A / denom
    r1 = _A * (1.0 - _E2) / denom ** 3
    d = x / (n1 * _K0)

    lat = phi1 - (n1 * tan_phi1 / r1) * (
        d ** 2 / 2.0
        - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1 ** 2 - 9.0 * _EP2) * d ** 4 / 24.0
        + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1 ** 2 - 252.0 * _EP2 - 3.0 * c1 ** 2) * d ** 6 / 720.0
    )
    lon_offset = (
        d
        - (1.0 + 2.0 * t1 + c1) * d ** 3 / 6.0
        + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1 ** 2 + 8.0 * _EP2 + 24.0 * t1 ** 2) * d ** 5 / 120.0
    ) / cos_phi1

    central_meridian = math.radians((zone - 1) * 6 - 180 + 3)
    return math.degrees(central_meridian + lon_offset), math.degrees(lat)


def lonlat_to_utm(lon: float, lat: float, zone: int, northern: bool = True) -> Tuple[float, float]:
    """Project a geographic coordinate onto UTM on the WGS84 ellipsoid.

    The inverse of :func:`utm_to_lonlat`, to the same series order.  Authoring a package
    needs this direction, because a household selected on the map is a longitude and a
    latitude while ``spawns.json`` records simulation coordinates.  ``sumolib`` can do it
    through ``net.convertLonLat2XY``, but only with ``pyproj`` installed, and it is not.
    """
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    central_meridian = math.radians((zone - 1) * 6 - 180 + 3)

    sin_phi = math.sin(lat_rad)
    cos_phi = math.cos(lat_rad)
    tan_phi = math.tan(lat_rad)

    n = _A / math.sqrt(1.0 - _E2 * sin_phi ** 2)
    t = tan_phi ** 2
    c = _EP2 * cos_phi ** 2
    a = cos_phi * (lon_rad - central_meridian)

    m = _A * (
        (1.0 - _E2 / 4.0 - 3.0 * _E2 ** 2 / 64.0 - 5.0 * _E2 ** 3 / 256.0) * lat_rad
        - (3.0 * _E2 / 8.0 + 3.0 * _E2 ** 2 / 32.0 + 45.0 * _E2 ** 3 / 1024.0) * math.sin(2.0 * lat_rad)
        + (15.0 * _E2 ** 2 / 256.0 + 45.0 * _E2 ** 3 / 1024.0) * math.sin(4.0 * lat_rad)
        - (35.0 * _E2 ** 3 / 3072.0) * math.sin(6.0 * lat_rad)
    )

    easting = _K0 * n * (
        a
        + (1.0 - t + c) * a ** 3 / 6.0
        + (5.0 - 18.0 * t + t ** 2 + 72.0 * c - 58.0 * _EP2) * a ** 5 / 120.0
    ) + _FALSE_EASTING

    northing = _K0 * (
        m
        + n * tan_phi * (
            a ** 2 / 2.0
            + (5.0 - t + 9.0 * c + 4.0 * c ** 2) * a ** 4 / 24.0
            + (61.0 - 58.0 * t + t ** 2 + 600.0 * c - 330.0 * _EP2) * a ** 6 / 720.0
        )
    )
    if not northern:
        northing += _FALSE_NORTHING
    return easting, northing


def build_inverse_projection(net: Any) -> Callable[[float, float], Tuple[float, float]]:
    """Return a converter from longitude and latitude into simulation coordinates.

    Mirrors :func:`build_projection`, which goes the other way.  ``sumolib``'s own
    converter is preferred when its projection backend is present, and the series
    implementation stands in when it is not.
    """
    try:
        if net.hasGeoProj():
            net.convertLonLat2XY(0.0, 0.0)

            def convert_sumolib(lon: float, lat: float) -> Tuple[float, float]:
                x, y = net.convertLonLat2XY(lon, lat)
                return float(x), float(y)

            return convert_sumolib
    except Exception:
        pass

    location = getattr(net, "_location", {}) or {}
    kind, zone, northern = _parse_proj(str(location.get("projParameter", "")))
    if kind == "none":
        raise ProjectionError("the network file records no geographic projection")
    if kind == "unsupported" or (kind == "utm" and zone is None):
        raise ProjectionError(
            f"unsupported projection {location.get('projParameter', '')!r}; "
            "install pyproj to let sumolib handle it"
        )

    offset = net.getLocationOffset()
    off_x, off_y = float(offset[0]), float(offset[1])

    if kind == "longlat":
        def convert_longlat(lon: float, lat: float) -> Tuple[float, float]:
            return lon + off_x, lat + off_y

        return convert_longlat

    def convert_utm(lon: float, lat: float) -> Tuple[float, float]:
        easting, northing = lonlat_to_utm(lon, lat, zone, northern)
        return easting + off_x, northing + off_y

    return convert_utm


def _sumolib_converter(net: Any) -> Optional[Callable[[float, float], Tuple[float, float]]]:
    """Return sumolib's own converter when its projection backend is present."""
    try:
        if not net.hasGeoProj():
            return None
        net.convertXY2LonLat(0.0, 0.0)
    except Exception:
        return None

    def convert(x: float, y: float) -> Tuple[float, float]:
        lon, lat = net.convertXY2LonLat(x, y)
        return float(lon), float(lat)

    return convert


def _utm_converter(net: Any) -> Callable[[float, float], Tuple[float, float]]:
    location = getattr(net, "_location", {}) or {}
    kind, zone, northern = _parse_proj(str(location.get("projParameter", "")))
    if kind == "none":
        raise ProjectionError("the network file records no geographic projection")
    if kind == "unsupported" or (kind == "utm" and zone is None):
        raise ProjectionError(
            f"unsupported projection {location.get('projParameter', '')!r}; "
            "install pyproj to let sumolib handle it"
        )

    offset = net.getLocationOffset()
    off_x, off_y = float(offset[0]), float(offset[1])

    if kind == "longlat":
        def convert_longlat(x: float, y: float) -> Tuple[float, float]:
            return x - off_x, y - off_y

        return convert_longlat

    def convert_utm(x: float, y: float) -> Tuple[float, float]:
        return utm_to_lonlat(x - off_x, y - off_y, zone, northern)

    return convert_utm


def build_projection(
    net: Any,
    *,
    verify: Optional[Callable[[float, float], Sequence[float]]] = None,
    sample_points: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[Callable[[float, float], Tuple[float, float]], dict]:
    """Return a converter for this network plus a short report on how it was built.

    Args:
        net: The ``sumolib`` net object the simulator already loaded.
        verify: Optional independent converter, normally ``traci.simulation.convertGeo``,
            used to confirm the result on sample points.
        sample_points: Simulation coordinates to check. Defaults to points spread
            across the network's own boundary.

    Raises:
        ProjectionError: If no usable projection exists, or if the converter and
            the verifier disagree by more than :data:`AGREEMENT_TOLERANCE_M`.
    """
    report: dict = {}
    convert = _sumolib_converter(net)
    report["source"] = "sumolib" if convert is not None else "builtin_utm"
    if convert is None:
        convert = _utm_converter(net)

    location = getattr(net, "_location", {}) or {}
    report["proj_parameter"] = str(location.get("projParameter", ""))

    if verify is None:
        report["verified"] = False
        return convert, report

    if sample_points is None:
        sample_points = _boundary_samples(location)

    worst = 0.0
    checked = 0
    for x, y in sample_points:
        try:
            expected = verify(x, y)
        except Exception:
            continue
        got = convert(x, y)
        worst = max(worst, _distance_m(got, (float(expected[0]), float(expected[1]))))
        checked += 1

    report["verified"] = checked > 0
    report["samples_checked"] = checked
    report["max_disagreement_m"] = round(worst, 3)
    if checked and worst > AGREEMENT_TOLERANCE_M:
        raise ProjectionError(
            f"coordinate conversion disagrees with SUMO by {worst:.1f} m, "
            f"which is too far to draw a map from"
        )
    return convert, report


def _boundary_samples(location: dict) -> List[Tuple[float, float]]:
    try:
        min_x, min_y, max_x, max_y = [float(v) for v in str(location["convBoundary"]).split(",")]
    except Exception:
        return [(0.0, 0.0), (1000.0, 1000.0)]
    mid_x = (min_x + max_x) / 2.0
    mid_y = (min_y + max_y) / 2.0
    return [
        (min_x, min_y),
        (max_x, max_y),
        (mid_x, mid_y),
        (min_x, max_y),
        (max_x, min_y),
    ]


def _distance_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great-circle distance between two longitude and latitude pairs, in metres."""
    lon1, lat1 = math.radians(a[0]), math.radians(a[1])
    lon2, lat2 = math.radians(b[0]), math.radians(b[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * 6371000.0 * math.asin(min(1.0, math.sqrt(h)))
