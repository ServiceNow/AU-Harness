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


def is_model_configured(model: str) -> bool:
    """Check if the model is configured.

    Args:
        model: the name of the model to check
    Returns:
        True if model is configured, False otherwise
    """
    if model_info == {}:
        load_model_info()
    return True if model_info.get(model) is not None else False


def add_model_info(name: str, new_model_info: dict) -> None:
    """Add a new model info or overwrite model info.

    Args:
        name: the name of the model to update info for
        new_model_info: dictionary containing model info
            {
              "url": "http://localhost:8181/generate",
              "auth_token": "Bearer AAAAAAAAAAAAAA",
              "parameters": {}
              "other":...
            }

    Returns:
        None
    """
    if model_info == {}:
        load_model_info()
    model_info[name] = new_model_info


def update_model_info(name: str, updated_model_info: dict[str, Any]) -> None:
    """Update model info if exists. Only update the properties provided in `updated_model_info`.

    Use values configured in config/models.json for the properties which are not provided in `updated_model_info`.

    Args:
        name: Unique name of the model
        updated_model_info: dictionary containing model info
    {
      "url": "http://localhost:8181/generate",
      "auth_token": "Bearer AAAAAAAAAAAAAA",
      "parameters": {}
      "other":...
    }

    Returns:
        None
    """
    if model_info == {}:
        load_model_info()
    model_info_for_model = model_info.get(name)
    assert model_info_for_model, f"Model {name} is not configured in config/models.json"

    for param, value in updated_model_info.items():
        if not isinstance(value, dict):
            model_info_for_model[param] = value
        else:
            # if it is json, update keys which are passed, do not remove other keys
            if param in model_info_for_model and isinstance(model_info_for_model[param], dict):
                model_info_for_model[param].update(value)
            else:
                model_info_for_model[param] = value


def update_api_key(model_info_for_model: dict, model: str) -> dict:
    """Update API Key based on the provided authentication type.

    Args:
        model_info_for_model (dict): A dictionary containing the model information,
        model (str): The model name.

    Returns:
        dict: The updated model information with the appropriate authentication token.

    Raises:
        KeyError: If 'auth_type' is missing from the model information.
        ValueError: If required fields for 'basic' authentication are missing.
    """

    def _check_token_type():
        if "," in model_info_for_model.get(constants.TOKEN_TYPE, ""):
            raise Exception("Separate token type with | instead of with a comma.")
        if "," in model_info_for_model.get(constants.AUTH_TOKEN, ""):
            raise Exception("Separate auth tokens with | instead of with a comma.")
        types = model_info_for_model[constants.TOKEN_TYPE].split("|")
        tokens = model_info_for_model[constants.AUTH_TOKEN].split("|")

        if len(types) != len(tokens):
            raise Exception(
                f"Number of token types does not match number of auth tokens for model {model}. If one of the endpoints does not have a token type, format as TYPE1||TYPE3."
            )

        converted = []
        for idx, token_type in enumerate(types):
            if token_type == constants.BEARER:
                converted.append(f"Bearer {tokens[idx]}")
            elif token_type == constants.BASIC:
                username = os.getenv(f"{env_prefix}_API_USERNAME")
                password = os.getenv(f"{env_prefix}_API_PASSWORD")
                if not username or not password:
                    raise ValueError(
                        f"{env_prefix}_API_USERNAME or {env_prefix}_API_PASSWORD is missing for the model: {model} in the "
                        f"environment variables for basic authentication."
                    )
                token = b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
                converted.append(f"Basic {token}")
        model_info_for_model["auth_token"] = "|".join(converted)

    env_prefix = f"{constants.CLAE}_{model}".upper()
    if not model_info_for_model.get(constants.AUTH_TOKEN):
        logger.warning(
            f"{env_prefix}_{constants.TOKEN.upper()} is missing for the model: {model} in the environment variables."
        )
        return model_info_for_model

    if not model_info_for_model.get(constants.TOKEN_TYPE):
        logger.warning(
            f"{env_prefix}_{constants.TOKEN_TYPE.upper()} is missing for the model: {model} in the environment "
            f"variables. Using the key as it is without any authentication."
        )
        return model_info_for_model

    _check_token_type()
    # make sure Bearer is not duplicated in auth token
    model_info_for_model["auth_token"] = re.sub(
        r"(?<=\s)Bearer",
        "",
        model_info_for_model["auth_token"],
        count=model_info_for_model["auth_token"].count("Bearer") - 1,
    ).strip()
    return model_info_for_model


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
