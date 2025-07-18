class Postprocessor():
    def process(self, dataset: list[dict], predictions, metric) -> tuple:
        raise NotImplementedError
