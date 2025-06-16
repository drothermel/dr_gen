from dr_gen.utils.utils import flatten_dict_tuple_keys

DEFAULT_XNAME = "epoch"
DEFAULT_METRIC_NAME = "loss"

# TODO: This could live in dr_util


class SplitMetrics:
    def __init__(
        self,
        config,
        split,
    ):
        self.config = config
        self.split = split

        self.curves = {}  # metric_name: MetricCurves

    def add_x_v(
        self,
        x,
        val,
        metric_name,
        x_name=DEFAULT_XNAME,
        x_val_hashable=True,
    ):
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

    def get_xs(self, metric_name=DEFAULT_METRIC_NAME, x_name=DEFAULT_XNAME):
        assert metric_name in self.curves, f">> {metric_name} not in curves"
        metric_curves_obj = self.curves[metric_name]
        return metric_curves_obj.get_xs(x_name=x_name)

    def get_vals(self, metric_name, x_name=DEFAULT_XNAME):
        assert metric_name in self.curves, f">> {metric_name} not in curves"
        metric_curves_obj = self.curves[metric_name]
        return metric_curves_obj.get_vals(x_name=x_name)

    def get_all_xs(self):
        xs = {}  # metric_name: x_name: list
        for metric_name, metric_curves in self.curves.items():
            xs[metric_name] = metric_curves.get_all_xs()
        return xs

    def get_all_xs_flat(self):
        nested_xs = self.get_all_xs()
        # (metric_name, x_name): list
        return flatten_dict_tuple_keys(nested_xs)

    def get_all_vals(self):
        vals = {}  # metric_name: x_name: list
        for metric_name, metric_curves in self.curves.items():
            vals[metric_name] = metric_curves.get_all_vals()
        return vals

    def get_all_vals_flat(self):
        nested_vals = self.get_all_vals()
        # (metric_name, x_name): list
        return flatten_dict_tuple_keys(nested_vals)

    def get_by_xval(self, xval, metric_name, x_name=DEFAULT_XNAME):
        assert metric_name in self.curves, f">> {metric_name} not in curves"
        metric_curves_obj = self.curves[metric_name]
        return metric_curves_obj.get_by_xval(xval, x_name=x_name)


class MetricCurves:
    def __init__(
        self,
        config,
        split,
        metric_name,
    ):
        self.config = config
        self.split = split
        self.metric_name = metric_name

        self.curves = {}  # x_name: MetricCurve

    def add_x_v(
        self,
        x,
        val,
        x_name=DEFAULT_XNAME,
        x_val_hashable=True,
    ):
        if x_name not in self.curves:
            self.curves[x_name] = MetricCurve(
                self.config,
                self.split,
                self.metric_name,
                x_name=x_name,
                x_val_hashable=x_val_hashable,
            )
        self.curves[x_name].add_x_v(x, val)

    def get_xs(self, x_name=DEFAULT_XNAME):
        assert x_name in self.curves, f">> {x_name} not in curves"
        return self.curves[x_name].xs

    def get_vals(self, x_name=DEFAULT_XNAME):
        assert x_name in self.curves, f">> {x_name} not in curves"
        return self.curves[x_name].vals

    def get_all_xs(self):
        xs = {}  # x_name: list
        for x_name, metric_curve in self.curves.items():
            xs[x_name] = metric_curve.xs
        return xs

    def get_all_vals(self):
        vals = {}  # x_name: list
        for x_name, metric_curve in self.curves.items():
            vals[x_name] = metric_curve.vals
        return vals

    def get_by_xval(self, xval, x_name=DEFAULT_XNAME):
        assert x_name in self.curves, f">> {x_name} not in curves"
        return self.curves[x_name].get_by_xval(xval)


class MetricCurve:
    def __init__(
        self,
        config,
        split,
        metric_name,
        x_name=DEFAULT_XNAME,
        x_val_hashable=True,
    ):
        self.config = config
        self.split = split
        self.metric_name = metric_name
        self.x_name = x_name
        self.x_val_hashable = x_val_hashable

        self.x_vals = []
        self.metric_vals = []
        self.x2met = {}

    @property
    def xs(self):
        return self.x_vals

    @property
    def vals(self):
        return self.metric_vals

    def add_x_v(self, x, val):
        if not self.x_val_hashable:
            x = str(x)
        assert x not in self.x2met, f">> {x} already exists"
        self.x_vals.append(x)
        self.metric_vals.append(val)
        self.x2met[x] = val

    def add_curve(self, xs, vals):
        assert len(self.x_vals) == 0, ">> x vals already exist"
        assert len(self.metric_vals) == 0, ">> metric vals already exist"
        assert len(self.x2met) == 0, ">> x2met already exists"
        assert len(xs) == len(vals), ">> xs and vals must be same length"
        self.x_vals = xs
        self.metric_vals = vals
        self.x2met = {x: m for x, m in zip(xs, vals, strict=False)}

    def sort_curve_by_x(self):
        assert len(self.x_vals) != 0, ">> there are no x vals"
        assert len(self.metric_vals) == len(self.x_vals), (
            ">> xs and vals must be same length"
        )
        combined = [(x, m) for x, m in zip(self.x_vals, self.metric_vals, strict=False)]
        after_sort = sorted(combined)
        self.x_vals = [x for x, _ in after_sort]
        self.metric_vals = [m for _, m in after_sort]

    def get_by_xval(self, xval):
        if not self.x_val_hashable:
            xval = str(xval)
        assert xval in self.x2met, f">> {xval} doesn't exist"
        return self.x2met[xval]
