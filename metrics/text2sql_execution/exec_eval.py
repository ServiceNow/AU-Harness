"""Code taken from https://github.com/taoyds/test-suite-sql-eval/blob/master/exec_eval.py file."""

import asyncio
import os
import random
import re
import sqlite3
import threading
from collections import defaultdict
from itertools import chain, product

import tqdm

import logging
logger = logging.getLogger(__name__)
logger.propagate = True

from metrics.text2sql_execution.parse import (
    get_all_preds_for_execution,
    remove_distinct,
)

threadLock = threading.Lock()
TIMEOUT = 60
EXEC_TMP_DIR = "tmp/"


def permute_tuple(element: tuple, perm: tuple) -> tuple:
    """Permute the elements of a tuple based on the given permutation.

    Args:
        element: The tuple to be permuted.
        perm: The permutation tuple.

    Returns:
        A new tuple with elements permuted according to perm.
    """
    assert len(element) == len(perm)
    return tuple([element[i] for i in perm])


def unorder_row(row: tuple) -> tuple:
    """Sort the elements of a row tuple.

    Args:
        row: The input row tuple.

    Returns:
        A new tuple with elements sorted.
    """
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))


def quick_rej(result1: list, result2: list, order_matters: bool) -> bool:
    """Quickly reject if two results are not equivalent.

    Args:
        result1: The first result list.
        result2: The second result list.
        order_matters: Whether the order of rows matters.

    Returns:
        True if the results might be equivalent, False if they are definitely not.
    """
    s1 = [unorder_row(row) for row in result1]
    s2 = [unorder_row(row) for row in result2]
    if order_matters:
        return s1 == s2
    else:
        return set(s1) == set(s2)


def multiset_eq(l1: list, l2: list) -> bool:
    """Check if two lists are equivalent as multisets.

    Args:
        l1: The first list.
        l2: The second list.

    Returns:
        True if the lists are equivalent as multisets, False otherwise.
    """
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] = d[e] + 1
    for e in l2:
        d[e] = d[e] - 1
        if d[e] < 0:
            return False
    return True


def get_constraint_permutation(tab1_sets_by_columns: list, result2: list):
    """Get constrained permutations for column matching.

    Args:
        tab1_sets_by_columns: List of sets containing unique values for each column in table 1.
        result2: The result from table 2.

    Returns:
        An iterator of possible permutations.
    """
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)

    for _ in range(20):
        random_tab2_row = random.choice(result2)

        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].remove(tab2_col)
    return product(*perm_constraints)


def result_eq(result1: list, result2: list, order_matters: bool) -> bool:
    """Check if two results are equivalent.

    Args:
        result1: The first result list.
        result2: The second result list.
        order_matters: Whether the order of rows matters.

    Returns:
        True if the results are equivalent, False otherwise.
    """
    if len(result1) == 0 and len(result2) == 0:
        return True

    if len(result1) != len(result2):
        return False

    num_cols = len(result1[0])

    if len(result2[0]) != num_cols:
        return False

    if not quick_rej(result1, result2, order_matters):
        return False

    tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]

    for perm in get_constraint_permutation(tab1_sets_by_columns, result2):
        if len(perm) != len(set(perm)):
            continue
        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [permute_tuple(element, perm) for element in result2]
        if order_matters:
            if result1 == result2_perm:
                return True
        else:
            if set(result1) == set(result2_perm) and multiset_eq(result1, result2_perm):
                return True
    return False


def replace_cur_year(query: str) -> str:
    """Replace YEAR(CURDATE()) with 2020 in the given query.

    Args:
        query: The input SQL query.

    Returns:
        The modified query with YEAR(CURDATE()) replaced.
    """
    return re.sub(
        "YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*", "2020", query, flags=re.IGNORECASE
    )


def get_cursor_from_path(sqlite_path: str):
    """Get a database cursor for the given SQLite database path.

    Args:
        sqlite_path: The path to the SQLite database.

    Returns:
        A cursor object for the database.
    """
    try:
        if not os.path.exists(sqlite_path):
            logger.info("Openning a new connection %s" % sqlite_path)
        connection = sqlite3.connect(sqlite_path)
    except Exception as e:
        logger.error(sqlite_path)
        raise e
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()
    return cursor


async def exec_on_db_(sqlite_path: str, query: str) -> tuple:
    """Execute a query on the specified database.

    Args:
        sqlite_path: The path to the SQLite database.
        query: The SQL query to execute.

    Returns:
        A tuple containing the execution status and result.
    """
    query = replace_cur_year(query)
    cursor = get_cursor_from_path(sqlite_path)
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        cursor.connection.close()
        return "result", result
    except Exception as e:
        cursor.close()
        cursor.connection.close()
        return "exception", e


async def exec_on_db(
    sqlite_path: str, query: str, process_id: str = "", timeout: int = TIMEOUT
) -> tuple:
    """Execute a query on the specified database with a timeout.

    Args:
        sqlite_path: The path to the SQLite database.
        query: The SQL query to execute.
        process_id: An optional process identifier.
        timeout: The maximum execution time in seconds.

    Returns:
        A tuple containing the execution status and result.
    """
    try:
        return await asyncio.wait_for(exec_on_db_(sqlite_path, query), timeout)
    except asyncio.TimeoutError:
        return ("exception", TimeoutError)
    except Exception as e:
        return ("exception", e)


def postprocess(query: str) -> str:
    """Postprocess the query to avoid execution errors.

    Args:
        query: The input SQL query.

    Returns:
        The postprocessed query.
    """
    query = query.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")
    return query


def eval_exec_match(
    db: str,
    p_str: str,
    g_str: str,
    plug_value: bool,
    keep_distinct: bool,
    progress_bar_for_each_datapoint: bool,
) -> int:
    """Evaluate if two queries are semantically equivalent by executing them on multiple databases.

    Args:
        db: The path to the main database.
        p_str: The predicted query string.
        g_str: The gold (correct) query string.
        plug_value: Whether to plug in values from the gold query to the predicted query.
        keep_distinct: Whether to keep DISTINCT in the queries.
        progress_bar_for_each_datapoint: Whether to show a progress bar for each datapoint.

    Returns:
        1 if the queries are semantically equivalent, 0 otherwise.
    """
    p_str, g_str = postprocess(p_str), postprocess(g_str)
    if not keep_distinct:
        p_str = remove_distinct(p_str)
        g_str = remove_distinct(g_str)

    order_matters = "order by" in g_str.lower()

    db_dir = os.path.dirname(db)
    db_paths = [
        os.path.join(db_dir, basename)
        for basename in os.listdir(db_dir)
        if ".sqlite" in basename
    ]

    preds = [p_str]
    if plug_value:
        _, preds = get_all_preds_for_execution(g_str, p_str)
        preds = chain([p_str], preds)

    for pred in preds:
        pred_passes = 1
        ranger = tqdm.tqdm(db_paths) if progress_bar_for_each_datapoint else db_paths

        for db_path in ranger:
            g_flag, g_denotation = asyncio.run(exec_on_db(db_path, g_str))
            p_flag, p_denotation = asyncio.run(exec_on_db(db_path, pred))

            assert (
                g_flag != "exception"
            ), f"gold query {g_str} has error on database file {db_path}"

            if p_flag == "exception":
                pred_passes = 0
            elif not result_eq(g_denotation, p_denotation, order_matters=order_matters):
                pred_passes = 0
            if pred_passes == 0:
                break

        if pred_passes == 1:
            return 1

    return 0