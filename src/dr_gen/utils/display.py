from typing import Any

from prettytable import PrettyTable


def make_table(fns, rows):
    """Create a PrettyTable with given field names and rows.

    Args:
        fns: Field names for the table.
        rows: Row data for the table.

    Returns:
        PrettyTable instance or None if no data.
    """
    if len(rows) == 0 or len(rows[0]) == 0:
        return None
    table = PrettyTable()
    table.field_names = fns
    if isinstance(rows[0][0], list):
        for rrs in rows:
            table.add_rows(rrs, divider=True)
    else:
        table.add_rows(rows)
    return table


def make_sort_key(table, sort_field_order):
    """Create a sort key function for table rows.

    Args:
        table: PrettyTable instance.
        sort_field_order: List of field names for sorting order.

    Returns:
        Sort key function.
    """
    # Create a mapping from field names to their corresponding index in
    #   table.field_names.
    field_index = {field: i for i, field in enumerate(table.field_names)}

    def sort_key(row) -> tuple:
        # The first element is the "sortby" field repeated
        row = row[1:]
        # Build a tuple of values for the row according to sort_field_order.
        return tuple(row[field_index[field]] for field in sort_field_order)

    return sort_key


def get_fields_for_drop_cols(table, cols):
    """Get field names excluding specified columns.

    Args:
        table: PrettyTable instance.
        cols: Columns to exclude.

    Returns:
        Set of remaining field names.
    """
    all_fns = set(table.field_names)
    return all_fns - set(cols)


def get_sortby_sort_key(table, sort_fields):
    """Get sortby field and sort key function.

    Args:
        table: PrettyTable instance.
        sort_fields: List of fields to sort by.

    Returns:
        Tuple of (sortby field, sort key function).
    """
    if len(sort_fields) == 0:
        return None, lambda x: x

    sortby = sort_fields[0]
    sort_key = make_sort_key(table, sort_fields)
    return sortby, sort_key


def get_filter_function(table, **kwargs: Any):  # noqa: ANN401
    """Create a filter function for table rows.

    Args:
        table: PrettyTable instance.
        **kwargs: Filter criteria.

    Returns:
        Filter function.
    """
    field_index = {field.lower(): i for i, field in enumerate(table.field_names)}

    def filter_function(vals) -> bool:
        for k, v in kwargs.items():
            k = k.lower()
            v = [v] if not isinstance(v, list) else v
            vstrs = [str(vv) for vv in v]
            ind = field_index.get(k)
            if ind is None:
                continue
            if vals[ind] not in vstrs:
                return False
        return True

    return filter_function


def print_drop_cols(table, cols):
    """Print table with specified columns dropped.

    Args:
        table: PrettyTable instance.
        cols: Columns to drop.
    """
    get_fields_for_drop_cols(table, cols)


def print_sorted(table, sort_fields):
    """Print table sorted by specified fields.

    Args:
        table: PrettyTable instance.
        sort_fields: Fields to sort by.
    """
    sortby, sort_key = get_sortby_sort_key(table, sort_fields)


def print_filtered(table, **kwargs: Any):  # noqa: ANN401
    """Print filtered table.

    Args:
        table: PrettyTable instance.
        **kwargs: Filter criteria.
    """
    get_filter_function(table, **kwargs)


def print_table(table, drop_cols=None, sort_cols=None, **filter_kwargs: Any):  # noqa: ANN401
    """Print table with optional column dropping, sorting, and filtering.

    Args:
        table: PrettyTable instance.
        drop_cols: Columns to drop from display.
        sort_cols: Columns to sort by.
        **filter_kwargs: Filter criteria.
    """
    if sort_cols is None:
        sort_cols = []
    if drop_cols is None:
        drop_cols = []
    fields_to_print = get_fields_for_drop_cols(table, drop_cols)
    sortby, sort_key = get_sortby_sort_key(table, sort_cols)
    filter_fxn = get_filter_function(table, **filter_kwargs)
    table.get_string(
        fields=fields_to_print,
        sortby=sortby,
        sort_key=sort_key,
        row_filter=filter_fxn,
    )
