
import os
import re
from typing import Any
from urllib.parse import urljoin

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from . import constants
from .constants import (
    MODEL_PARAM_MAPPER_FILE,
)
from .json_util import load_json_file

RE_TOOLKIT_JOB_URL = re.compile(r"^(?:https://)?(?P<job_id>[a-z0-9_-]+)\.job\.console\.elementai\.com(?![^/])")
RE_URL_PROTOCOL = re.compile(r"^https?://")



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


def _get_common_prompts(model_type: str = "default") -> dict:
    """Get common prompts stored in config/prompts/common_prompts.yaml.

    Use case: If the dataset file (JSON) has a common prompt in inputs_pretokenized, it will first be replaced from this file.
    As Jinja2 does not support partial replacement, we simply replace the variable {{variable_name}} with the value.

    Args:
    ----
        model_type: The model type

    Returns:
    -------
        The common prompt dictionary
    """
    common_prompt_file = "config/prompts/common_prompts.yaml"
    with open(common_prompt_file) as f:
        common_prompts_all = yaml.safe_load(f)
    return common_prompts_all.get("prompt_templates", {}).get(model_type, {})


def get_common_prompts(model_type: str = "default") -> dict:
    """Get common prompt dictionary.

    Key will be in the form of prompt_templates.default.<actualKey>
    Args:
        model_type: The model type

    Returns:
    -------
        The common prompt dictionary

    """
    tmpl_dict = _get_common_prompts(model_type)
    # build the full path in the key, in the form of common_prompts.default.<actual_variable_key>
    return {f"prompt_templates.{model_type}.{k}": v for k, v in tmpl_dict.items()}


def get_language_prompt_template(template_name: str = "default") -> dict:
    """Get language prompt template."""
    common_prompt_file = "config/prompts/language_prompts.yaml"
    with open(common_prompt_file) as f:
        common_prompts_all = yaml.safe_load(f).get(template_name)
    return common_prompts_all



def model_param_name_mapper(model_name: str, parameter_name: str, inference_type="default") -> str:
    """Build key mapper dictionary. If the model key is missing, the model parameter dictionary will be None.

    Parameter names that are not mapped will be returned as is, only containing mapped parameters;
    undefined parameters will be ignored.

    The mapping works as follows
    * if model_name.inference_type exists, return the mapping
    * if all.inference_type exists, return the mapping
    * if no model_name, return parameter name as is
    * return model_name.default

    Args:
    ----
        model_name: The model name
        parameter_name: The parameter name
        inference_type: The inference type

    Returns:
    -------
        The mapped parameter name
    """
    param_mapper = {}
    cfg_paths = [constants.CLAE_ROOT_PATH, constants.CLAE_INTERNAL_PATH]
    for cfgpath in cfg_paths:
        if Path(cfgpath + MODEL_PARAM_MAPPER_FILE).exists():
            param_mapper.update(load_json_file(cfgpath + MODEL_PARAM_MAPPER_FILE))

    # get model param mapper based on the key
    model_param_map = None
    for model_key in param_mapper.keys():
        # if model_key partially matches with mapper key, return dictionary
        if model_key in model_name:
            model_param_map = param_mapper.get(model_key)

    if (model_param_map is not None) and (
        inference_type is not None and model_param_map.get(inference_type) is not None
    ):
        return model_param_map.get(inference_type).get(parameter_name)
    elif inference_type is not None and param_mapper.get("all").get(inference_type) is not None:
        return param_mapper.get("all").get(inference_type).get(parameter_name)
    elif model_param_map is None:
        return parameter_name
    else:
        return model_param_map.get("default").get(parameter_name)


def model_param_mapper(model_name: str, params: dict, inference_type: str | None = "default") -> dict:
    """Map the model parameters based on the model name.

    Args:
    ----
    model_name: The model name
    params: The parameters
    inference_type: The inference type

    Returns:
        The mapped parameters

    """
    if not inference_type:
        inference_type = "default"
    final_params = {}
    for k, v in params.items():
        pk = model_param_name_mapper(model_name, k, inference_type)
        # if param name found, add into the new dict
        if pk is not None and final_params.get(pk) is None:
            final_params[pk] = v
    return final_params

def add_cache_url(*, cache_url: str, url: str) -> str:
    """Add the cache URL to the given URL.

    Args:
        cache_url: The URL of the caching proxy.
        url: The URL to add the cache URL to, typically the URL of a model.

    Returns:
        The URL modified to go through the caching proxy.
        If `cache_url` is empty, or if `url.startswith(cache_url)`, then `url` is returned as is.
    """
    if (match := RE_TOOLKIT_JOB_URL.match(cache_url)) and not RE_TOOLKIT_JOB_URL.match(url):
        suggestion = (
            "an internal DNS URL"
            if "EAI_CONSOLE_URL" in os.environ  # If CLAE is running on Toolkit.
            else f"http://localhost:8080/job/{match['job_id']} while running `eai proxy` in a different terminal"
        )
        logger.warning(
            f"The cache URL {cache_url} is a Toolkit job access URL, but the model URL {url} is not. "
            "Toolkit authentication may be missing in order to reach the caching proxy. "
            f"If so, consider disabling caching (`CLAE_MODEL_CONFIG__CACHE_URL=`) or using a different cache URL. Maybe {suggestion}?"
        )

    # The added slash ensures the URL is appended to the cache URL and urljoin() normalizes consecutive slashes.
    return url if url.startswith(cache_url) else urljoin(cache_url + "/", RE_URL_PROTOCOL.sub("", url))



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



