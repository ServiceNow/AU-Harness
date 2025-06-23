import ast
import itertools
import json
import re
from collections.abc import Generator

import regex

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

RE_JSON_START = re.compile(r"[{[]")
TYPE_FIELD = "type"
RE_JSON_UNESCAPED_BACKSLASH = re.compile(
    r"""
        # Match an empty string where a backslash should be inserted to escape an invalid unescaped backslash.
        (?<!\\)  # Don't match if the backslash is already escaped with a backslash.
        # Match an empty string
        (?=  # if it is followed
            \\  # by a backslash.
            (?!  # Don't match if the backslash is part of a valid JSON escape sequence
                ["/bfnrt]  # \" \/ \b \f \n \r \t
                |  # or
                u[0-9A-Fa-f]{4}  # \uXXXX
                |  # or
                \\(?:\\\\)*(?!\\)  # \\, \\\\, \\\\\\, etc. (an even number of backslashes)
            )
        )
    """,
    re.VERBOSE,
)
RE_JSON_UNESCAPED_DOUBLE_QUOTE = regex.compile(
    r"""
        # Match an empty string where a backslash should be inserted to escape an invalid unescaped double quote.
        # This pattern requires the `regex` library since the standard `re` library doesn't support variable-length lookbehind.
        (?<!  # Don't match if the double quote is preceded
            [,:[{]\s*  # by any of `,:[{`, optionally with whitespaces in between (the double quote is presumably valid)
            |  # or
            (?<!\\)\\  # by exactly one backslash (the double quote is already escaped).
        )
        # Match an empty string
        (?=  # if it is followed
            "  # by a double quote.
            (?!  # Don't match if the double quote is followed
                \s*[,:\]}]  # by any of `,:]}` (the double quote is presumably valid).
            )
        )
    """,
    regex.VERBOSE,
)


def _fix_backslashes(input_text: str):
    characters = "_+-[]*'$"
    double_backslash_pattern = rf"\\\\([{re.escape(characters)}])"
    single_backslash_pattern = rf"\\([{re.escape(characters)}])"
    input_text = re.sub(double_backslash_pattern, r"\1", input_text)
    input_text = re.sub(single_backslash_pattern, r"\1", input_text)
    return input_text


def _fix_json_unescaped_backslashes(text: str) -> str:
    return RE_JSON_UNESCAPED_BACKSLASH.sub(r"\\", text)


def _fix_json_unescaped_double_quotes(text: str) -> str:
    return RE_JSON_UNESCAPED_DOUBLE_QUOTE.sub(r"\\", text)


def dump_json_file(filepath: str, data, ensure_ascii=False, indent=4, sort_keys=True):
    """Dump data to a JSON file.

    Args:
    ----
        filepath: The file path to the JSON file.
        data: The data to dump.
        ensure_ascii: See `json.dump()`.
        indent: See `json.dump()`.
        sort_keys: See `json.dump()`.
        **kwargs: Additional arguments to pass to `json.dump()`.
    """
    with open(filepath, "w") as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys)
        f.write("\n")


def load_json_file(filepath: str) -> dict | list:
    """Method to load a JSON file.

    Args:
    ----
        filepath: The file path

    Returns:
    -------
        The JSON object

    """
    with open(filepath) as f:
        return json.load(f)


def load_jsonl_file(filepath: str, nrows: int | None = None) -> list[dict | list]:
    """Method to load a JSONL file.

    Args:
    ----
        filepath: The file path
        nrows: Stop after that many rows. If None, read all rows.

    Returns:
    -------
        The JSON object

    """
    with open(filepath, "r") as f:
        data = [json.loads(line) for line in itertools.islice(f, nrows)]
    return data


def extract_and_load_json_iter(
    text: str, *, start: int = 0, strict: bool = False
) -> Generator[tuple[dict | list | None, str], None, None]:
    """Method to extract JSON objects and arrays from text, even if there is text around.

    Args:
    ----
        text: The text
        start: Start searching at this index
        strict: Whether to use strict JSON decoding

    Returns:
    -------
        Generator that yields all valid JSON objects and arrays found in the text, as well as the text that was matched.
        If there is no valid JSON object in the text, the generator will be empty.

    """
    decoder = json.JSONDecoder(strict=strict)
    while match := RE_JSON_START.search(text, start):
        start = match.start()
        try:
            json_object, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start += 1
        else:
            yield json_object, text[start:end]
            start = end


def extract_and_load_json(
    text: str, ignore_backslashes: bool = True, strict: bool = False, return_all: bool | None = None
) -> dict | list | None:
    """Method to extract and load JSON from text.

    Args:
    ----
        text: str: The text
        ignore_backslashes: bool: Whether to remove backslashes. Both options are available so
            we can compare the results with and without removing backslashes.
        strict: Whether to use strict JSON decoding
        return_all: bool | None: If True, return all matching JSON objects as a list.
            If False or None, return the first match.

    Returns:
    -------
        dict: The JSON object or None

    """
    # TODO Improve and copy what we do in production
    if ignore_backslashes:
        text = _fix_backslashes(text)
        text = _fix_json_unescaped_double_quotes(text)
        text = _fix_json_unescaped_backslashes(text)

    if return_all:
        json_objects = [json_object for json_object, _ in extract_and_load_json_iter(text, strict=strict)]
        if not json_objects:
            logger.error(f"Error extracting multiple JSON from text: {text}.")
        return json_objects

    json_object, _ = next(extract_and_load_json_iter(text, strict=strict), (None, ""))
    if json_object is None:
        logger.error(f"Error extracting JSON from text: {text}.")
    return json_object


def extract_and_load_json_ast(text: str) -> dict | list | None:
    """Method to extract and load JSON-like objects from text using ast.

    The ast module can load most JSON, even if it uses `'` instead of `"`,
    but it fails if the JSON contains `true`, `false`, or `null`.

    Args:
    ----
        text: str: The text

    Returns:
    -------
        dict: The JSON object or None

    """
    json_object = None
    try:
        # extract JSON from surrounded text
        text = re.findall(r"[\[\n\r\s]*{.+}[\n\r\s\]]*", text, re.DOTALL)[0]
        json_object = ast.literal_eval(text)
    except Exception as e:
        logger.error(f"Error extracting JSON from text: {text}. Exception: {e}")
    return json_object


def load_json(text: str) -> dict | list | None:
    """Method to load JSON from text.

    Args:
    ----
        text: str : The text

    Returns:
    -------
        dict: The JSON object or None

    """
    json_object = None
    try:
        json_object = json.loads(text)
    except Exception as e:
        logger.error(f"Error extracting JSON from text: {text}. Exception: {e}")
    return json_object


def is_valid_json(s: str) -> bool:
    """Check if a string is a valid JSON."""
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


def is_minified(
    s: str, *, allow_spaces_in_separators: bool = False, allow_newlines: bool = False, strict: bool = False
) -> bool:
    """Check if a string is minified JSON, i.e. the smallest possible JSON string.

    Args:
    ----
        s: JSON string. If the string is not valid JSON, a JSONDecodeError will be raised.
            You can call is_valid_json first to avoid the error.
        allow_spaces_in_separators: If True, the JSON can contain spaces after separators (":", ",").
        allow_newlines: If True, the JSON can contain newlines.
        strict: Whether to use strict JSON decoding

    Returns:
    -------
        bool: True if the JSON string is minified as specified, False otherwise
    """
    # The null character `\0` is used in temporary `separators` before being replaced with the regex pattern ` ?`.
    separators = (",\0", ":\0") if allow_spaces_in_separators else (",", ":")
    indent = "" if allow_newlines else None

    minified = json.dumps(json.loads(s, strict=strict), ensure_ascii=False, indent=indent, separators=separators)
    return re.fullmatch(re.escape(minified).replace("\0", " ?").replace("\n", "\n?"), s) is not None


def is_pretty_printed(s: str) -> bool:
    """Check if a string is pretty printed JSON.

    The indent can be a tab or any number of spaces, but it has to be consistent throughout the JSON string.

    Args:
        s: JSON string. If the string is not valid JSON, a JSONDecodeError will be raised.
            You can call is_valid_json first to avoid the error.
    """
    # The null character `\0` is used as temporary `indent` in `json.dumps()` before being replaced with:
    # - a capturing group for the first indent; and
    # - a backreference for all remaining indents, ensuring they are all identical to the first one.
    re_pretty = (
        re.escape(json.dumps(json.loads(s), indent="\0", ensure_ascii=False))
        .replace("\0", r"(\t| +)", 1)
        .replace("\0", r"\1")
    )
    return re.fullmatch(re_pretty, s) is not None
