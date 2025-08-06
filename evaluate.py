import argparse
import asyncio
import logging
import json
# Apply nest_asyncio to allow nested event loops in Azure OpenAI client calls
import nest_asyncio

from utils.engine import create_engine, run_all_engines
from utils.model_utils import register_models_with_controller
from utils.util import read_config, expand_dataset_metric_pairs, _calculate_aggregates

nest_asyncio.apply()

# Module-level logger - will be configured when setup_logging is called
logger = logging.getLogger(__name__)


def main(cfg_path='config.yaml'):
    """Main function to run the evaluation benchmark.

    Args:
        cfg_path: Path to configuration file

    Returns:
        Dictionary containing evaluation scores
    """
    # 1. Read config and process configuration dictionaries
    cfg, judge_properties, filters, temperature_overrides, aggregates = read_config(cfg_path)
    # 2. Initialize central request controller
    central_request_controller, model_configs = register_models_with_controller(cfg.get("models", []), judge_properties)

    # 3. Expand dataset-metric pairs using runspecs
    expanded_pairs = expand_dataset_metric_pairs(cfg)

    # 4. Create engines for each expanded dataset-metric pair
    all_engines = []
    for dataset_name, metric_name, dataset_info, task_type in expanded_pairs:
        engine, dataset_name = create_engine(
            dataset_name=dataset_name,
            dataset_info=dataset_info,
            task_type=task_type,
            metric_name=metric_name,
            filters=filters,
            model_configs=model_configs,
            temperature_overrides=temperature_overrides,
            judge_properties=judge_properties,
            central_request_controller=central_request_controller
        )
        all_engines.append((engine, dataset_name))

    # 5. Run all engines concurrently
    scores = asyncio.run(run_all_engines(all_engines))

    # 6. Log final results and process aggregates
    if aggregates:
        _calculate_aggregates(aggregates, scores, model_configs)

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run audio evaluation benchmark')
    parser.add_argument('--config', '-c', default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    args = parser.parse_args()

    # Pass the config path to main
    all_scores = main(cfg_path=args.config)
    logger.info("[main] Evaluation complete. Final results:")
    logger.info(json.dumps(all_scores, indent=2))
