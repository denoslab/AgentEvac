"""Unit tests for scripts.plot_run_metrics."""

from scripts.plot_run_metrics import _kpi_specs, _plot_kpi_grid


class TestKpiSpecs:
    def test_extracts_expected_values(self):
        metrics = {
            "departure_time_variability": 59.2975,
            "route_choice_entropy": 0.686473,
            "average_hazard_exposure": {"global_average": 0.0},
            "average_travel_time": {"average": 599.4855},
        }
        specs = _kpi_specs(metrics)
        values = {str(item["title"]): float(item["value"]) for item in specs}
        assert values["Departure variance"] == 59.2975
        assert values["Route entropy"] == 0.686473
        assert values["Hazard exposure"] == 0.0
        assert values["Avg travel time"] == 599.4855

    def test_missing_fields_default_to_zero(self):
        specs = _kpi_specs({})
        assert all(float(item["value"]) == 0.0 for item in specs)


class TestPlotMetricsDashboard:
    class _FakeAxis:
        def __init__(self):
            self.ylabel = None
            self.ylim = None
            self.title = None
            self.text_calls = []

        def bar(self, *args, **kwargs):
            return None

        def set_title(self, value, **kwargs):
            self.title = value

        def set_ylabel(self, value, **kwargs):
            self.ylabel = value

        def set_xticks(self, *args, **kwargs):
            return None

        def set_ylim(self, *args, **kwargs):
            self.ylim = args

        def grid(self, *args, **kwargs):
            return None

        def text(self, *args, **kwargs):
            self.text_calls.append((args, kwargs))

    class _FakeSubGrid:
        def __getitem__(self, key):
            return key

    class _FakeSlot:
        def subgridspec(self, *args, **kwargs):
            return TestPlotMetricsDashboard._FakeSubGrid()

    class _FakeFigure:
        def __init__(self):
            self.axes = []

        def add_subplot(self, _slot):
            ax = TestPlotMetricsDashboard._FakeAxis()
            self.axes.append(ax)
            return ax

    def test_plot_kpi_grid_creates_four_separate_panels(self):
        metrics = {
            "departure_time_variability": 59.2975,
            "route_choice_entropy": 0.686473,
            "average_hazard_exposure": {"global_average": 0.0},
            "average_travel_time": {"average": 599.4855},
        }
        fig = self._FakeFigure()
        slot = self._FakeSlot()

        _plot_kpi_grid(fig, slot, metrics)

        assert len(fig.axes) == 4
        assert [ax.title for ax in fig.axes] == [
            "Departure variance",
            "Route entropy",
            "Hazard exposure",
            "Avg travel time",
        ]
        assert [ax.ylabel for ax in fig.axes] == [
            "Seconds^2",
            "Entropy (nats)",
            "Average risk score",
            "Seconds",
        ]
        assert all(ax.ylim is not None for ax in fig.axes)
