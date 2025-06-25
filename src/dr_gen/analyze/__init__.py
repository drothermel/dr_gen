"""Analyze package for dr_gen - utilities for analyzing training runs."""


def check_prefix_exclude(check_string: str, excluded_prefixes: list[str]) -> bool:
    """Check if string starts with any of the excluded prefixes.

    Args:
        check_string: String to check
        excluded_prefixes: List of prefix strings to check against

    Returns:
        True if check_string starts with any excluded prefix, False otherwise
    """
    return any(check_string.startswith(pre) for pre in excluded_prefixes)
