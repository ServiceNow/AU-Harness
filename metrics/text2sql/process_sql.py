"""Code taken from https://github.com/taoyds/test-suite-sql-eval/blob/master/process_sql.py file."""

import json
import sqlite3

from nltk import word_tokenize

CLAUSE_KEYWORDS = (
    "select",
    "from",
    "where",
    "group",
    "order",
    "limit",
    "intersect",
    "union",
    "except",
)
JOIN_KEYWORDS = ("join", "on", "as")

WHERE_OPS = (
    "not",
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "in",
    "like",
    "is",
    "exists",
)
UNIT_OPS = ("none", "-", "+", "*", "/")
AGG_OPS = ("none", "max", "min", "count", "sum", "avg")
TABLE_TYPE = {
    "sql": "sql",
    "table_unit": "table_unit",
}

COND_OPS = ("and", "or")
SQL_OPS = ("intersect", "union", "except")
ORDER_OPS = ("desc", "asc")


class Schema:
    """Simple schema which maps table and column to a unique identifier.

    Attributes:
        _schema (dict[str, list[str]]): The schema dictionary.
        _idMap (dict[str, str]): The mapping of table and column to unique identifiers.
    """

    def __init__(self, schema: dict[str, list[str]]):
        """Initialize the Schema object.

        Args:
            schema: A dictionary representing the database schema.
        """
        self._schema = schema
        self._idMap = self._map(self._schema)

    @property
    def schema(self) -> dict[str, list[str]]:
        """Get the schema dictionary.

        Returns:
            The schema dictionary.
        """
        return self._schema

    @property
    def idMap(self) -> dict[str, str]:
        """Get the id mapping dictionary.

        Returns:
            The id mapping dictionary.
        """
        return self._idMap

    def _map(self, schema: dict[str, list[str]]) -> dict[str, str]:
        """Create a mapping of table and column names to unique identifiers.

        Args:
            schema: A dictionary representing the database schema.

        Returns:
            A dictionary mapping table and column names to unique identifiers.
        """
        idMap = {"*": "__all__"}
        count = 1
        for key, vals in schema.items():
            for val in vals:
                idMap[key.lower() + "." + val.lower()] = (
                    f"__{key.lower()}.{val.lower()}__"
                )
                count += 1

        for key in schema:
            idMap[key.lower()] = f"__{key.lower()}__"
            count += 1

        return idMap


def get_schema(db: str) -> dict[str, list[str]]:
    """Get database's schema, which is a dict with table name as key and list of column names as value.

    Args:
        db: Database path.

    Returns:
        Schema dict where keys are table names and values are lists of column names.
    """
    schema = {}
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # fetch table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(table[0].lower()) for table in cursor.fetchall()]

    # fetch table info
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]

    return schema


def get_schema_from_json(fpath: str) -> dict[str, list[str]]:
    """Get schema from a JSON file.

    Args:
        fpath: File path to the JSON schema file.

    Returns:
        Schema dict where keys are table names and values are lists of column names.
    """
    with open(fpath) as f:
        data = json.load(f)

    schema = {}
    for entry in data:
        table = str(entry["table"].lower())
        cols = [str(col["column_name"].lower()) for col in entry["col_data"]]
        schema[table] = cols

    return schema


def tokenize(string: str) -> list[str]:
    """Tokenize the input string.

    Args:
        string: The input string to tokenize.

    Returns:
        A list of tokens.
    """
    string = str(string)
    string = string.replace(
        "'", '"'
    )  # ensures all string values wrapped by "" problem??
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"

    # keep string value as token
    vals = {}
    for i in range(len(quote_idxs) - 1, -1, -2):
        qidx1 = quote_idxs[i - 1]
        qidx2 = quote_idxs[i]
        val = string[qidx1 : qidx2 + 1]
        key = f"__val_{qidx1}_{qidx2}__"
        string = string[:qidx1] + key + string[qidx2 + 1 :]
        vals[key] = val

    toks = [word.lower() for word in word_tokenize(string)]
    # replace with string value token
    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]

    # find if there exists !=, >=, <=
    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    eq_idxs.reverse()
    prefix = ("!", ">", "<")
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx - 1]
        if pre_tok in prefix:
            toks = toks[: eq_idx - 1] + [pre_tok + "="] + toks[eq_idx + 1 :]

    return toks


def scan_alias(toks: list[str]) -> dict[str, str]:
    """Scan the index of 'as' and build the map for all aliases.

    Args:
        toks: A list of tokens.

    Returns:
        A dictionary mapping aliases to their original names.
    """
    as_idxs = [idx for idx, tok in enumerate(toks) if tok == "as"]
    alias = {}
    for idx in as_idxs:
        alias[toks[idx + 1]] = toks[idx - 1]
    return alias


def get_tables_with_alias(
    schema: dict[str, list[str]], toks: list[str]
) -> dict[str, str]:
    """Get a dictionary of tables and their aliases.

    Args:
        schema: The database schema.
        toks: A list of tokens.

    Returns:
        A dictionary mapping table aliases (and original names) to table names.
    """
    tables = scan_alias(toks)
    for key in schema:
        assert key not in tables, f"Alias {key} has the same name in table"
        tables[key] = key
    return tables


def parse_col(
    toks: list[str],
    start_idx: int,
    tables_with_alias: dict[str, str],
    schema: Schema,
    default_tables: list[str] | None = None,
) -> tuple[int, str]:
    """Parse a column from the token list.

    Args:
        toks: A list of tokens.
        start_idx: The starting index in the token list.
        tables_with_alias: A dictionary of tables and their aliases.
        schema: The database schema.
        default_tables: A list of default table names.

    Returns:
        A tuple containing the next index and the column id.
    """
    tok = toks[start_idx]
    if tok == "*":
        return start_idx + 1, schema.idMap[tok]

    if "." in tok:  # if token is a composite
        alias, col = tok.split(".")
        key = tables_with_alias[alias] + "." + col
        return start_idx + 1, schema.idMap[key]

    assert (
        default_tables is not None and len(default_tables) > 0
    ), "Default tables should not be None or empty"

    for alias in default_tables:
        table = tables_with_alias[alias]
        if tok in schema.schema[table]:
            key = table + "." + tok
            return start_idx + 1, schema.idMap[key]

    raise AssertionError(f"Error col: {tok}")


def parse_col_unit(
    toks: list[str],
    start_idx: int,
    tables_with_alias: dict[str, str],
    schema: Schema,
    default_tables: list[str] | None = None,
) -> tuple[int, tuple[int, str, bool]]:
    """Parse a column unit from the token list.

    Args:
        toks: A list of tokens.
        start_idx: The starting index in the token list.
        tables_with_alias: A dictionary of tables and their aliases.
        schema: The database schema.
        default_tables: A list of default table names.

    Returns:
        A tuple containing the next index and a tuple of (agg_op id, col_id, isDistinct).
    """
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    isDistinct = False
    if toks[idx] == "(":
        isBlock = True
        idx += 1

    if toks[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(toks[idx])
        idx += 1
        assert idx < len_ and toks[idx] == "("
        idx += 1
        if toks[idx] == "distinct":
            idx += 1
            isDistinct = True
        idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)
        assert idx < len_ and toks[idx] == ")"
        idx += 1
        return idx, (agg_id, col_id, isDistinct)

    if toks[idx] == "distinct":
        idx += 1
        isDistinct = True
    agg_id = AGG_OPS.index("none")
    idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ")"
        idx += 1  # skip ')'

    return idx, (agg_id, col_id, isDistinct)


def parse_val_unit(
    toks: list[str],
    start_idx: int,
    tables_with_alias: dict[str, str],
    schema: Schema,
    default_tables: list[str] | None = None,
) -> tuple[int, tuple[int, tuple[int, str, bool], tuple[int, str, bool] | None]]:
    """Parse a value unit from the token list.

    Args:
        toks: A list of tokens.
        start_idx: The starting index in the token list.
        tables_with_alias: A dictionary of tables and their aliases.
        schema: The database schema.
        default_tables: A list of default table names.

    Returns:
        A tuple containing the next index and a tuple of (unit_op, col_unit1, col_unit2).
    """
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == "(":
        isBlock = True
        idx += 1

    col_unit1 = None
    col_unit2 = None
    unit_op = UNIT_OPS.index("none")

    idx, col_unit1 = parse_col_unit(
        toks, idx, tables_with_alias, schema, default_tables
    )
    if idx < len_ and toks[idx] in UNIT_OPS:
        unit_op = UNIT_OPS.index(toks[idx])
        idx += 1
        idx, col_unit2 = parse_col_unit(
            toks, idx, tables_with_alias, schema, default_tables
        )

    if isBlock:
        assert toks[idx] == ")"
        idx += 1  # skip ')'

    return idx, (unit_op, col_unit1, col_unit2)


def parse_table_unit(
    toks: list[str], start_idx: int, tables_with_alias: dict[str, str], schema: Schema
) -> tuple[int, str, str]:
    """Parse a table unit from the token list.

    Args:
        toks: A list of tokens.
        start_idx: The starting index in the token list.
        tables_with_alias: A dictionary of tables and their aliases.
        schema: The database schema.

    Returns:
        A tuple containing the next index, table id, and table name.
    """
    idx = start_idx
    len_ = len(toks)
    key = tables_with_alias[toks[idx]]

    if idx + 1 < len_ and toks[idx + 1] == "as":
        idx += 3
    else:
        idx += 1

    return idx, schema.idMap[key], key


def parse_value(
    toks: list[str],
    start_idx: int,
    tables_with_alias: dict[str, str],
    schema: Schema,
    default_tables: list[str] | None = None,
) -> tuple[int, str | float | dict]:
    """Parse a value from the token list.

    Args:
        toks: A list of tokens.
        start_idx: The starting index in the token list.
        tables_with_alias: A dictionary of tables and their aliases.
        schema: The database schema.
        default_tables: A list of default table names.

    Returns:
        A tuple containing the next index and the parsed value.
    """
    idx = start_idx
    len_ = len(toks)

    isBlock = False
    if toks[idx] == "(":
        isBlock = True
        idx += 1

    if toks[idx] == "select":
        idx, val = parse_sql(toks, idx, tables_with_alias, schema)
    elif '"' in toks[idx]:  # token is a string value
        val = toks[idx]
        idx += 1
    else:
        try:
            val = float(toks[idx])
            idx += 1
        except Exception:
            end_idx = idx
            while (
                end_idx < len_
                and toks[end_idx] != ","
                and toks[end_idx] != ")"
                and toks[end_idx] != "and"
                and toks[end_idx] not in CLAUSE_KEYWORDS
                and toks[end_idx] not in JOIN_KEYWORDS
            ):
                end_idx += 1

            idx, val = parse_col_unit(
                toks[start_idx:end_idx], 0, tables_with_alias, schema, default_tables
            )
            idx = end_idx

    if isBlock:
        assert toks[idx] == ")"
        idx += 1

    return idx, val


def parse_condition(
    toks: list[str],
    start_idx: int,
    tables_with_alias: dict[str, str],
    schema: Schema,
    default_tables: list[str] | None = None,
) -> tuple[int, list]:
    """Parse a condition from the token list.

    Args:
        toks: A list of tokens.
        start_idx: The starting index in the token list.
        tables_with_alias: A dictionary of tables and their aliases.
        schema: The database schema.
        default_tables: A list of default table names.

    Returns:
        A tuple containing the next index and the parsed condition.
    """
    idx = start_idx
    len_ = len(toks)
    conds = []

    while idx < len_:
        idx, val_unit = parse_val_unit(
            toks, idx, tables_with_alias, schema, default_tables
        )
        not_op = False
        if toks[idx] == "not":
            not_op = True
            idx += 1

        assert (
            idx < len_ and toks[idx] in WHERE_OPS
        ), f"Error condition: idx: {idx}, tok: {toks[idx]}"
        op_id = WHERE_OPS.index(toks[idx])
        idx += 1
        val1 = val2 = None
        if op_id == WHERE_OPS.index(
            "between"
        ):  # between..and... special case: dual values
            idx, val1 = parse_value(
                toks, idx, tables_with_alias, schema, default_tables
            )
            assert toks[idx] == "and"
            idx += 1
            idx, val2 = parse_value(
                toks, idx, tables_with_alias, schema, default_tables
            )
        else:  # normal case: single value
            idx, val1 = parse_value(
                toks, idx, tables_with_alias, schema, default_tables
            )
            val2 = None

        conds.append((not_op, op_id, val_unit, val1, val2))

        if idx < len_ and (
            toks[idx] in CLAUSE_KEYWORDS
            or toks[idx] in (")", ";")
            or toks[idx] in JOIN_KEYWORDS
        ):
            break

        if idx < len_ and toks[idx] in COND_OPS:
            conds.append(toks[idx])
            idx += 1  # skip and/or

    return idx, conds


def parse_select(
    toks: list[str],
    start_idx: int,
    tables_with_alias: dict[str, str],
    schema: Schema,
    default_tables: list[str] | None = None,
) -> tuple[int, tuple[bool, list]]:
    """Parse a SELECT statement from the token list.

    Args:
        toks: A list of tokens.
        start_idx: The starting index in the token list.
        tables_with_alias: A dictionary of tables and their aliases.
        schema: The database schema.
        default_tables: A list of default table names.

    Returns:
        A tuple containing the next index and a tuple of (isDistinct, list of value units).
    """
    idx = start_idx
    len_ = len(toks)

    assert toks[idx] == "select", "'select' not found"
    idx += 1
    isDistinct = False
    if idx < len_ and toks[idx] == "distinct":
        idx += 1
        isDistinct = True
    val_units = []

    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
        idx, val_unit = parse_val_unit(
            toks, idx, tables_with_alias, schema, default_tables
        )
        val_units.append((agg_id, val_unit))
        if idx < len_ and toks[idx] == ",":
            idx += 1  # skip ','

    return idx, (isDistinct, val_units)


def parse_from(
    toks: list[str], start_idx: int, tables_with_alias: dict[str, str], schema: Schema
) -> tuple[int, dict, list[str]]:
    """Parse a FROM clause from the token list.

    Args:
        toks: A list of tokens.
        start_idx: The starting index in the token list.
        tables_with_alias: A dictionary of tables and their aliases.
        schema: The database schema.

    Returns:
        A tuple containing the next index, a dictionary of table units and conditions, and a list of default tables.
    """
    assert "from" in toks[start_idx:], "'from' not found"

    len_ = len(toks)
    idx = toks.index("from", start_idx) + 1
    default_tables = []
    table_units = []
    conds = []

    while idx < len_:
        isBlock = False
        if toks[idx] == "(":
            isBlock = True
            idx += 1

        if toks[idx] == "select":
            idx, sql = parse_sql(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE["sql"], sql))
        else:
            if idx < len_ and toks[idx] == "join":
                idx += 1  # skip join
            idx, table_unit, table_name = parse_table_unit(
                toks, idx, tables_with_alias, schema
            )
            table_units.append((TABLE_TYPE["table_unit"], table_unit))
            default_tables.append(table_name)
        if idx < len_ and toks[idx] == "on":
            idx += 1  # skip on
            idx, this_conds = parse_condition(
                toks, idx, tables_with_alias, schema, default_tables
            )
            if len(conds) > 0:
                conds.append("and")
            conds.extend(this_conds)

        if isBlock:
            assert toks[idx] == ")"
            idx += 1
        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            break

    return idx, table_units, conds, default_tables


def parse_where(
    toks: list[str],
    start_idx: int,
    tables_with_alias: dict[str, str],
    schema: Schema,
    default_tables: list[str],
) -> tuple[int, list]:
    """Parse a WHERE clause from the token list.

    Args:
        toks: A list of tokens.
        start_idx: The starting index in the token list.
        tables_with_alias: A dictionary of tables and their aliases.
        schema: The database schema.
        default_tables: A list of default table names.

    Returns:
        A tuple containing the next index and the parsed conditions.
    """
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != "where":
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_group_by(
    toks: list[str],
    start_idx: int,
    tables_with_alias: dict[str, str],
    schema: Schema,
    default_tables: list[str],
) -> tuple[int, list]:
    """Parse a GROUP BY clause from the token list.

    Args:
        toks: A list of tokens.
        start_idx: The starting index in the token list.
        tables_with_alias: A dictionary of tables and their aliases.
        schema: The database schema.
        default_tables: A list of default table names.

    Returns:
        A tuple containing the next index and the parsed column units.
    """
    idx = start_idx
    len_ = len(toks)
    col_units = []

    if idx >= len_ or toks[idx] != "group":
        return idx, col_units

    idx += 1
    assert toks[idx] == "by"
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(
            toks, idx, tables_with_alias, schema, default_tables
        )
        col_units.append(col_unit)
        if idx < len_ and toks[idx] == ",":
            idx += 1  # skip ','
        else:
            break

    return idx, col_units


def parse_order_by(
    toks: list[str],
    start_idx: int,
    tables_with_alias: dict[str, str],
    schema: Schema,
    default_tables: list[str],
) -> tuple[int, tuple[str, list]]:
    """Parse an ORDER BY clause from the token list.

    Args:
        toks: A list of tokens.
        start_idx: The starting index in the token list.
        tables_with_alias: A dictionary of tables and their aliases.
        schema: The database schema.
        default_tables: A list of default table names.

    Returns:
        A tuple containing the next index and a tuple of (order_type, list of value units).
    """
    idx = start_idx
    len_ = len(toks)
    val_units = []
    order_type = "asc"  # default type is 'asc'

    if idx >= len_ or toks[idx] != "order":
        return idx, val_units

    idx += 1
    assert toks[idx] == "by"
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, val_unit = parse_val_unit(
            toks, idx, tables_with_alias, schema, default_tables
        )
        val_units.append(val_unit)
        if idx < len_ and toks[idx] in ORDER_OPS:
            order_type = toks[idx]
            idx += 1
        if idx < len_ and toks[idx] == ",":
            idx += 1  # skip ','
        else:
            break

    return idx, (order_type, val_units)


def parse_having(
    toks: list[str],
    start_idx: int,
    tables_with_alias: dict[str, str],
    schema: Schema,
    default_tables: list[str],
) -> tuple[int, list]:
    """Parse a HAVING clause from the token list.

    Args:
        toks: A list of tokens.
        start_idx: The starting index in the token list.
        tables_with_alias: A dictionary of tables and their aliases.
        schema: The database schema.
        default_tables: A list of default table names.

    Returns:
        A tuple containing the next index and the parsed conditions.
    """
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != "having":
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_limit(toks: list[str], start_idx: int) -> tuple[int, int | None]:
    """Parse a LIMIT clause from the token list.

    Args:
        toks: A list of tokens.
        start_idx: The starting index in the token list.

    Returns:
        A tuple containing the next index and the limit value (or None if not present).
    """
    idx = start_idx
    len_ = len(toks)

    if idx < len_ and toks[idx] == "limit":
        idx += 2
        # make limit value can work, cannot assume put 1 as a fake limit number
        if not toks[idx - 1].isdigit():
            return idx, 1

        return idx, int(toks[idx - 1])

    return idx, None


def parse_sql(
    toks: list[str], start_idx: int, tables_with_alias: dict[str, str], schema: Schema
) -> tuple[int, dict]:
    """Parse a complete SQL query from the token list.

    Args:
        toks: A list of tokens.
        start_idx: The starting index in the token list.
        tables_with_alias: A dictionary of tables and their aliases.
        schema: The database schema.

    Returns:
        A tuple containing the next index and a dictionary representing the parsed SQL query.
    """
    isBlock = False  # indicate whether this is a block of sql/sub-sql
    len_ = len(toks)
    idx = start_idx

    sql = {}
    if toks[idx] == "(":
        isBlock = True
        idx += 1

    # parse from clause in order to get default tables
    from_end_idx, table_units, conds, default_tables = parse_from(
        toks, start_idx, tables_with_alias, schema
    )
    sql["from"] = {"table_units": table_units, "conds": conds}
    # select clause
    _, select_col_units = parse_select(
        toks, idx, tables_with_alias, schema, default_tables
    )
    idx = from_end_idx
    sql["select"] = select_col_units
    # where clause
    idx, where_conds = parse_where(toks, idx, tables_with_alias, schema, default_tables)
    sql["where"] = where_conds
    # group by clause
    idx, group_col_units = parse_group_by(
        toks, idx, tables_with_alias, schema, default_tables
    )
    sql["groupBy"] = group_col_units
    # having clause
    idx, having_conds = parse_having(
        toks, idx, tables_with_alias, schema, default_tables
    )
    sql["having"] = having_conds
    # order by clause
    idx, order_col_units = parse_order_by(
        toks, idx, tables_with_alias, schema, default_tables
    )
    sql["orderBy"] = order_col_units
    # limit clause
    idx, limit_val = parse_limit(toks, idx)
    sql["limit"] = limit_val

    idx = skip_semicolon(toks, idx)
    if isBlock:
        assert toks[idx] == ")"
        idx += 1  # skip ')'
    idx = skip_semicolon(toks, idx)

    # intersect/union/except clause
    for op in SQL_OPS:  # initialize IUE
        sql[op] = None
    if idx < len_ and toks[idx] in SQL_OPS:
        sql_op = toks[idx]
        idx += 1
        idx, IUE_sql = parse_sql(toks, idx, tables_with_alias, schema)
        sql[sql_op] = IUE_sql
    return idx, sql


def load_data(fpath: str) -> dict:
    """Load data from a JSON file.

    Args:
        fpath: File path to the JSON data file.

    Returns:
        The loaded data as a dictionary.
    """
    with open(fpath) as f:
        data = json.load(f)
    return data


def get_sql(schema: Schema, query: str) -> dict:
    """Get the parsed SQL structure from a query string.

    Args:
        schema: The database schema.
        query: The SQL query string.

    Returns:
        A dictionary representing the parsed SQL query.
    """
    toks = tokenize(query)
    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    _, sql = parse_sql(toks, 0, tables_with_alias, schema)

    return sql


def skip_semicolon(toks: list[str], start_idx: int) -> int:
    """Skip semicolons in the token list.

    Args:
        toks: A list of tokens.
        start_idx: The starting index in the token list.

    Returns:
        The index after skipping semicolons.
    """
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx