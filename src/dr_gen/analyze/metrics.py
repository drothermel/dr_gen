from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from dr_gen.analyze.parsing import Hpm

DEFAULT_XNAME = "epoch"
DEFAULT_METRIC_NAME = "loss"

# TODO: This could live in dr_util


def _flatten_dict_tuple_keys(
    d: dict[Any, Any], parent_key: tuple[Any, ...] = ()
) -> dict[tuple[Any, ...], Any]:
    """Recursively flatten a nested dictionary with tuple keys."""
    items = {}
    for k, v in d.items():
        new_key = (*parent_key, k)
        if isinstance(v, dict):
            items.update(_flatten_dict_tuple_keys(v, new_key))
        else:
            items[new_key] = v
    return items


class SplitMetrics:
    """Manages metric curves for a specific split (train/val/eval)."""

    def __init__(
        self,
        config: "Hpm | DictConfig | dict[str, Any]",
        split: str,
    ) -> None:
        """Initialize SplitMetrics for a given config and split."""
        self.config = config
        self.split = split

        self.curves = {}  # metric_name: MetricCurves

    def add_x_v(
        self,
        x: float | str,
        val: float,
        metric_name: str,
        x_name: str = DEFAULT_XNAME,
        x_val_hashable: bool = True,
    ) -> None:
        """Add an x-value pair to the specified metric curve."""
        if metric_name not in self.curves:
            self.curves[metric_name] = MetricCurves(
                self.config,
                self.split,
                metric_name,
            )
        self.curves[metric_name].add_x_v(
            x,
            val,
            x_name=x_name,
            x_val_hashable=x_val_hashable,
        )

    def get_xs(
        self, metric_name: str = DEFAULT_METRIC_NAME, x_name: str = DEFAULT_XNAME
    ) -> list[int | float | str]:
        """Get x-values for a specific metric and x-axis name."""
        assert metric_name in self.curves, f">> {metric_name} not in curves"
        metric_curves_obj = self.curves[metric_name]
        return metric_curves_obj.get_xs(x_name=x_name)

    def get_vals(self, metric_name: str, x_name: str = DEFAULT_XNAME) -> list[float]:
        """Get metric values for a specific metric and x-axis name."""
        assert metric_name in self.curves, f">> {metric_name} not in curves"
        metric_curves_obj = self.curves[metric_name]
        return metric_curves_obj.get_vals(x_name=x_name)

    def get_all_xs(self):
        """Get all x-values organized by metric name and x-axis name."""
        xs = {}  # metric_name: x_name: list
        for metric_name, metric_curves in self.curves.items():
            xs[metric_name] = metric_curves.get_all_xs()
        return xs

    def get_all_xs_flat(self):
        """Get all x-values as a flattened dictionary with tuple keys."""
        nested_xs = self.get_all_xs()
        # (metric_name, x_name): list
        return _flatten_dict_tuple_keys(nested_xs)

    def get_all_vals(self):
        """Get all metric values organized by metric name and x-axis name."""
        vals = {}  # metric_name: x_name: list
        for metric_name, metric_curves in self.curves.items():
            vals[metric_name] = metric_curves.get_all_vals()
        return vals

    def get_all_vals_flat(self):
        """Get all metric values as a flattened dictionary with tuple keys."""
        nested_vals = self.get_all_vals()
        # (metric_name, x_name): list
        return _flatten_dict_tuple_keys(nested_vals)

    def get_by_xval(self, xval, metric_name, x_name=DEFAULT_XNAME):
        """Get metric value at a specific x-value for a given metric."""
        assert metric_name in self.curves, f">> {metric_name} not in curves"
        metric_curves_obj = self.curves[metric_name]
        return metric_curves_obj.get_by_xval(xval, x_name=x_name)


class MetricCurves:
    """Manages multiple curves for a specific metric."""

    def __init__(
        self,
        config: "Hpm | DictConfig | dict[str, Any]",
        split: str,
        metric_name: str,
    ) -> None:
        """Initialize MetricCurves for a specific metric."""
        self.config = config
        self.split = split
        self.metric_name = metric_name

        self.curves = {}  # x_name: MetricCurve

    def add_x_v(
        self,
        x: float | str,
        val: float,
        x_name: str = DEFAULT_XNAME,
        x_val_hashable: bool = True,
    ) -> None:
        """Add an x-value and metric value pair to the specified curve."""
        if x_name not in self.curves:
            self.curves[x_name] = MetricCurve(
                self.config,
                self.split,
                self.metric_name,
                x_name=x_name,
                x_val_hashable=x_val_hashable,
            )
        self.curves[x_name].add_x_v(x, val)

    def get_xs(self, x_name: str = DEFAULT_XNAME) -> list[int | float | str]:
        """Get x-values for a specific x-axis name."""
        assert x_name in self.curves, f">> {x_name} not in curves"
        return self.curves[x_name].xs

    def get_vals(self, x_name: str = DEFAULT_XNAME) -> list[float]:
        """Get metric values for a specific x-axis name."""
        assert x_name in self.curves, f">> {x_name} not in curves"
        return self.curves[x_name].vals

    def get_all_xs(self) -> dict[str, list[int | float | str]]:
        """Get all x-values organized by x-axis name."""
        xs = {}  # x_name: list
        for x_name, metric_curve in self.curves.items():
            xs[x_name] = metric_curve.xs
        return xs

    def get_all_vals(self) -> dict[str, list[float]]:
        """Get all metric values organized by x-axis name."""
        vals = {}  # x_name: list
        for x_name, metric_curve in self.curves.items():
            vals[x_name] = metric_curve.vals
        return vals

    def get_by_xval(self, xval: float | str, x_name: str = DEFAULT_XNAME) -> float:
        """Get metric value at a specific x-value for a given x-axis name."""
        assert x_name in self.curves, f">> {x_name} not in curves"
        return self.curves[x_name].get_by_xval(xval)


class MetricCurve:
    """Represents a single metric curve with x-values and metric values."""

    def __init__(
        self,
        config: "Hpm | DictConfig | dict[str, Any]",
        split: str,
        metric_name: str,
        x_name: str = DEFAULT_XNAME,
        x_val_hashable: bool = True,
    ) -> None:
        """Initialize MetricCurve for a single metric curve."""
        self.config = config
        self.split = split
        self.metric_name = metric_name
        self.x_name = x_name
        self.x_val_hashable = x_val_hashable

        self.x_vals = []
        self.metric_vals = []
        self.x2met = {}

    @property
    def xs(self) -> list[int | float | str]:
        """Get x-values for this curve."""
        return self.x_vals

    @property
    def vals(self) -> list[float]:
        """Get metric values for this curve."""
        return self.metric_vals

    def add_x_v(self, x: float | str, val: float) -> None:
        """Add a single x-value and metric value pair to this curve."""
        if not self.x_val_hashable:
            x = str(x)
        assert x not in self.x2met, f">> {x} already exists"
        self.x_vals.append(x)
        self.metric_vals.append(val)
        self.x2met[x] = val

    def add_curve(self, xs: list[int | float | str], vals: list[float]) -> None:
        """Add entire lists of x-values and metric values to this curve."""
        assert len(self.x_vals) == 0, ">> x vals already exist"
        assert len(self.metric_vals) == 0, ">> metric vals already exist"
        assert len(self.x2met) == 0, ">> x2met already exists"
        assert len(xs) == len(vals), ">> xs and vals must be same length"
        self.x_vals = xs
        self.metric_vals = vals
        self.x2met = dict(zip(xs, vals, strict=False))

    def sort_curve_by_x(self) -> None:
        """Sort the curve data by x-values in ascending order."""
        assert len(self.x_vals) != 0, ">> there are no x vals"
        assert len(self.metric_vals) == len(self.x_vals), (
            ">> xs and vals must be same length"
        )
        combined = [(x, m) for x, m in zip(self.x_vals, self.metric_vals, strict=False)]
        after_sort = sorted(combined)
        self.x_vals = [x for x, _ in after_sort]
        self.metric_vals = [m for _, m in after_sort]

    def get_by_xval(self, xval: float | str) -> float:
        """Get metric value at a specific x-value."""
        if not self.x_val_hashable:
            xval = str(xval)
        assert xval in self.x2met, f">> {xval} doesn't exist"
        return self.x2met[xval]
