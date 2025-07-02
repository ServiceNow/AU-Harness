
from typing import Any

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from . import constants

def smart_round(val: float, precision: int = constants.ROUND_DIGITS) -> float:
    """Round off metrics to global precision value.

    References:
        1. https://bugs.python.org/msg358467
        2. https://en.wikipedia.org/wiki/IEEE_754

    Args:
    ----
        precision: int: Precision up to which value should be rounded off.
        val: float: Value

    Returns:
    -------
        float: Rounded off value
    """
    if not isinstance(val, (int, float)):
        logger.error(f"Invalid value type: {type(val)}. Expected int or float.")
        raise ValueError("Invalid value passed, cannot be rounded off.")
    elif not isinstance(precision, int) or precision <= 0:
        logger.warning(
            f"Invalid precision provided: {precision}. Using the default precision: {constants.ROUND_DIGITS}"
        )
        precision = constants.ROUND_DIGITS
    rounded_off_val = round(val * 10**precision) / 10**precision
    return rounded_off_val

def get_context_indices_for_filter(key: str, value: Any, contexts: list[dict]) -> list[int]:
    """Get indices for rows satisfying the given filter.

    Given key-value pair, it returns the list of indices of contexts satisfying key = value.

    Args:
        key: The key to match against in each row of context/data.
        value: The value to compare against
        contexts: list of dictionaries containing additional key-value pairs in data.

    Returns:
        List of integer indices.

    """
    indices = [_ for _, c in enumerate(contexts) if c[key] == value]
    return indices



