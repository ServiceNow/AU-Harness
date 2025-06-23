from models.model import Model


class CustomModel(Model):
    """This class can be used to define a custom model in models.json.

    Define URL, auth_token, inference_type and formatter.
    For huggingface models, chat_formatter and formatter might contain the model id.

    Once done with the model configuration, define model_param_mapper.json
    While designing runspecs, we define parameters to be used by the test model.
    However, some parameters might be unavailable in some models or inference servers.
    Map the required parameters and skip the ones that are not available.
    """

    def name(self):
        """TODO: Need SME to add."""
        return self.model_name

    def __init__(self, model: str, model_info: dict):
        """TODO: Need SME to add."""
        self.model_name = model
        super().__init__(model_info)
