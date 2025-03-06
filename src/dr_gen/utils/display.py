from prettytable import PrettyTable


def make_table(fns, rows):
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
    # Create a mapping from field names to their corresponding index in
    #   table.field_names.
    field_index = {field: i for i, field in enumerate(table.field_names)}

    def sort_key(row):
        # The first element is the "sortby" field repeated
        row = row[1:]
        # Build a tuple of values for the row according to sort_field_order.
        return tuple(row[field_index[field]] for field in sort_field_order)

    return sort_key


def get_fields_for_drop_cols(table, cols):
    all_fns = set(table.field_names)
    print_fns = all_fns - set(cols)
    return print_fns


def get_sortby_sort_key(table, sort_fields):
    if len(sort_fields) == 0:
        return None, lambda x: x

    sortby = sort_fields[0]
    sort_key = make_sort_key(table, sort_fields)
    return sortby, sort_key


def get_filter_function(table, **kwargs):
    field_index = {field.lower(): i for i, field in enumerate(table.field_names)}

    def filter_function(vals):
        for k, v in kwargs.items():
            k = k.lower()
            v = [v] if not isinstance(v, list) else v
            vstrs = [str(vv) for vv in v]
            ind = field_index.get(k, None)
            if ind is None:
                continue
            if vals[ind] not in vstrs:
                return False
        return True

    return filter_function


def print_drop_cols(table, cols):
    print_fns = get_fields_for_drop_cols(table, cols)
    print(table.get_string(fields=print_fns))


def print_sorted(table, sort_fields):
    sortby, sort_key = get_sortby_sort_key(table, sort_fields)
    print(table.get_string(sortby=sortby, sort_key=sort_key))


def print_filtered(table, **kwargs):
    filter_function = get_filter_function(table, **kwargs)
    print(table.get_string(row_filter=filter_function))


def print_table(table, drop_cols=[], sort_cols=[], **filter_kwargs):
    fields_to_print = get_fields_for_drop_cols(table, drop_cols)
    sortby, sort_key = get_sortby_sort_key(table, sort_cols)
    filter_fxn = get_filter_function(table, **filter_kwargs)
    table_str = table.get_string(
        fields=fields_to_print,
        sortby=sortby,
        sort_key=sort_key,
        row_filter=filter_fxn,
    )
    print(table_str)
