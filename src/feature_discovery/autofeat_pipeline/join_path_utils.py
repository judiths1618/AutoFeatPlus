def get_path_length(path: str) -> int:
    # Path looks like table_source/table_name/key--table_source...
    path_tokens = path.split("--")
    # Length = 1 means that we have 2 tables
    return len(path_tokens) - 1


_SEP = "|"  # field separator within a join step (safe: never appears in CSV paths/column names)


def compute_join_name(join_key_property: tuple, partial_join_name: str) -> str:
    """
    Compute the name of the partial join, given the properties of the new join and the previous join name.

    :param join_key_property: (neo4j relation property, outbound label, inbound label)
    :param partial_join_name: Name of the partial join.
    :return: The name of the next partial join

    Format: {partial}--{from_table}|{from_col}|{to_col}|{to_table}
    Using '|' as field separator (never appears in file paths or column names),
    avoiding ambiguity with '-' in filenames like amf-performance_seg01_20211110.csv.
    """
    join_prop, from_table, to_table = join_key_property
    step = f"{from_table}{_SEP}{join_prop['from_column']}{_SEP}{join_prop['to_column']}{_SEP}{to_table}"
    return f"{partial_join_name}--{step}"


def parse_join_step(step: str) -> tuple:
    """Parse a single join step produced by compute_join_name.

    Returns (from_table, from_col, to_col, to_table).
    """
    parts = step.split(_SEP)
    if len(parts) != 4:
        raise ValueError(f"Malformed join step (expected 4 '|'-separated fields): {step!r}")
    return tuple(parts)
