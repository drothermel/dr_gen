from prettytable import PrettyTable


def make_table(fns, rows):
    table = PrettyTable()
    table.field_names = fns
    if isinstance(rows[0][0], list):
        for rrs in rows:
            table.add_rows(rrs, divider=True)
    else:
        table.add_rows(rows)
    return table
