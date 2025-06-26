import copy
import json
import os
import re
from base64 import b64encode
from typing import Any

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from utils import constants

# store the model information to reuse
# lazy initialization
model_info: dict[str, Any] = {}


def reset_model_info():
    """Clear global cache for model_info.

    Returns:
        None
    """
    global model_info
    model_info = {}


def load_model_info():
    """Load the model info from config file and store in global cache.

    Returns:
        None
    """
    cfg_paths = [constants.CLAE_ROOT_PATH, constants.CLAE_INTERNAL_PATH]
    for cfgpath in cfg_paths:
        # load the model info and default properties from conf file
        try:
            with open(cfgpath + constants.MODEL_INFO_FILE, "r") as f:
                # override external model config with internal(high priority), or add if the model key are new
                model_info.update(json.load(f))
        except FileNotFoundError:
            pass


def get_model_info(model: str) -> dict:
    """Get the model info for specific model from global cache.

    Load it from config file if not initialized

    Args:
        model: the model name to get info for

    Returns:
        copy of model info for model if found, else ValueError is thrown
    """
    enforce_check = os.getenv(f"{constants.CLAE}_ENFORCE_MODEL_CONFIG_CHECK", "True").lower() == "true"
    if model_info == {}:
        load_model_info()

    model_info_for_model = model_info.get(model)
    if model_info_for_model is None:
        raise Exception(f"Model {model} not defined in the model configuration file.")

    # Define the keys to check in .env
    env_prefix = f"{constants.CLAE}_{model}".upper()
    keys_to_check = [constants.URL, constants.TOKEN, constants.TOKEN_TYPE, constants.WEIGHTS] + list(
        model_info_for_model.keys()
    )
    if model.startswith(constants.MODEL_OPEN_ROUTER):
        env_prefix = f"{constants.CLAE}_{constants.MODEL_OPEN_ROUTER}".upper()

    # Update model_info_for_model with .env values if present
    # overwriting the values from the models.json config file
    for key in keys_to_check:
        env_var = os.getenv(f"{env_prefix}_{key}".upper())
        if env_var:
            dict_key = key if key != constants.TOKEN else constants.AUTH_TOKEN
            model_info_for_model[dict_key] = env_var
            logger.debug(f"Updated {dict_key} for model {model} from environment variables.")

    if (name := f"{env_prefix}_{constants.SEED}".upper()) in os.environ:
        if seed_value := os.environ[name]:
            seed = int(seed_value)
            model_info_for_model.setdefault("parameters", {})["seed"] = seed
            logger.info(f"Seed value updated to {seed} for model {model}.")
        else:
            model_info_for_model.get("parameters", {}).pop("seed", None)
            logger.info(f"Seed value removed for model {model}.")
    else:
        default_seed = model_info_for_model.get("parameters", {}).get("seed")
        logger.info(f"Using default seed {default_seed} for model {model}.") if default_seed else logger.info(
            f"Seed value not present for model {model}."
        )

    # Check if all required fields are present after .env update
    required_keys = [constants.URL] if enforce_check else []
    for key in required_keys:
        if model != "jury":
            if key not in model_info_for_model or not model_info_for_model[key]:
                key = key.upper() if key != constants.AUTH_TOKEN else constants.TOKEN.upper()
                raise Exception(
                    f"Required configuration for {key} is missing for model {model}.",
                    f"Add it to .env or environment variables following the {env_prefix}_key format; where key is URL, TOKEN etc.",
                )

    # Update the auth_token based on the auth_type
    model_info_for_model = update_api_key(model_info_for_model, model)
    return copy.deepcopy(model_info_for_model)


def override_model_params(original_model_params: dict, overridden_model_params: dict) -> dict:
    """Overrides model params.

    Args:
        original_model_params: current model params
        overridden_model_params: updated model_params to override with

    Returns:
        updated model_params if there are overridden model params
    """
    if (
        constants.CLI_MODEL_PARAM_FORCE_USE in original_model_params
        and original_model_params[constants.CLI_MODEL_PARAM_FORCE_USE]
    ):
        logger.info(
            f"Model params {original_model_params} are set to apply all the time. Ignoring all model params specified in models.json, runspec configs, and chain."
        )
        return original_model_params

    updated_model_params = overridden_model_params if overridden_model_params else original_model_params

    # Handle "stop" parameter in overridden model params
    if overridden_model_params and "stop" in overridden_model_params:
        logger.warning(f"Overridden model params contain a stop token param: {overridden_model_params['stop']}.")

    # Append original stop tokens if they exist
    if "stop" in original_model_params:
        updated_model_params["stop"] = list(set(updated_model_params.get("stop", []) + original_model_params["stop"]))

    for param in constants.KEEP_PARAMS:
        if param in original_model_params:
            updated_model_params.setdefault(param, original_model_params[param])

    if overridden_model_params:
        logger.info(f"Model params updated from {original_model_params} to {updated_model_params}")

    return updated_model_params
