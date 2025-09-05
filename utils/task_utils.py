import sys
import pathlib
import copy
import logging
from urllib.parse import urlparse
from ruamel.yaml import YAML

logger = logging.getLogger(__name__)

yaml = YAML(typ="safe")

def deep_merge(a, b):
    """
    Recursively merge two dictionaries with proper deep copying.
    
    Values from dictionary 'b' overwrite values from dictionary 'a' when keys conflict.
    For nested dictionaries, the merging continues recursively.
    
    Args:
        a: First dictionary to merge
        b: Second dictionary to merge, with higher precedence
        
    Returns:
        A new dictionary containing merged values from both input dictionaries
    """
    if isinstance(a, dict) and isinstance(b, dict):
        out = {k: deep_merge(a.get(k), v) if k in a else copy.deepcopy(v) for k, v in b.items()}
        for k, v in a.items():
            if k not in b:
                out[k] = copy.deepcopy(v)
        return out
    return copy.deepcopy(b)

def load_yaml(path):
    """
    Load and parse a YAML file.
    
    Args:
        path: Path to the YAML file to load
        
    Returns:
        The parsed YAML content as Python objects
    """
    with open(path, "r") as f:
        return yaml.load(f)

def json_pointer_get(doc, pointer):
    """
    Retrieve a value from a document using JSON Pointer syntax.
    
    Args:
        doc: The document (dict) to retrieve the value from
        pointer: JSON Pointer string (e.g., "#/path/to/item")
        
    Returns:
        The value at the specified path in the document
        
    Raises:
        AssertionError: If the pointer format is invalid
    """
    if not pointer or pointer == "#":
        return doc
    assert pointer.startswith("#/"), f"Bad pointer: {pointer}"
    cur = doc
    for raw in pointer[2:].split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        cur = cur[key]
    return cur

def resolve_ref(base_file, ref, seen):
    """
    Resolve a reference to another config file and retrieve the referenced node.
    
    Args:
        base_file: Path to the file containing the reference
        ref: Reference string in the format 'path/to/file#fragment'
        seen: Set of already processed references to detect cycles
        
    Returns:
        The resolved configuration node
        
    Raises:
        ValueError: If a cyclic reference is detected
    """
    url = urlparse(ref)
    ref_path = (pathlib.Path(base_file).parent / (url.path or base_file)).resolve()
    key = f"{ref_path}#{url.fragment or ''}"
    if key in seen:
        raise ValueError(f"Cyclic extends detected at {key}")
    seen.add(key)

    doc = load_yaml(ref_path)
    node = json_pointer_get(doc, url.fragment or "#")

    if isinstance(node, dict) and node.get("extends"):
        acc = {}
        for parent in node["extends"]:
            acc = deep_merge(acc, resolve_ref(ref_path, parent, seen))
        leaf = {k: v for k, v in node.items() if k != "extends"}
        node = deep_merge(acc, leaf)

    return node

def resolve_file(file_path):
    """
    Resolve a task config with extends references.

    Args:
        file_path: Path to the task config file

    Returns:
        Resolved config
    """
    doc = load_yaml(file_path)
    acc = {}
    for parent in (doc.get("extends") or []):
        acc = deep_merge(acc, resolve_ref(file_path, parent, set()))
    leaf = {k: v for k, v in doc.items() if k != "extends"}
    return deep_merge(acc, leaf)

def get_all_configs_in_dir(dir: pathlib.Path) -> list[pathlib.Path]:
    yaml_files = list(dir.glob("**/*.yaml"))
    yml_files = list(dir.glob("**/*.yml"))
    return [f for f in yaml_files if f.name !="base.yaml"] + [f for f in yml_files if f.name !="base.yaml"]

def build_task_ancestry(tasks_dir: pathlib.Path, task_configs: dict) -> dict:
    """
    Build task ancestry information by tracking the complete folder structure for each task.

    Args:
        tasks_dir: Base directory of all tasks
        task_configs: Dictionary mapping directory paths to task config files

    Returns:
        Dictionary mapping task names to their ancestry paths (list including base_dir, parent directories, and task name)
    """
    task_ancestry = {}

    # Process each group and its tasks
    for group_path, task_config_list in task_configs.items():
        for task_config in task_config_list:
            # Get task name from the resolved config
            task_name = task_config.get('task_name')
            if not task_name:
                logger.warning(f"Task config missing task_name: {task_config}")
                continue

            # Get relative path from base directory
            rel_path = group_path.relative_to(tasks_dir)
            
            # Get base directory name
            base_dir_name = tasks_dir.name
            
            # Build ancestry as list of path parts
            # Start with the base directory, then add intermediate directories, and finally the task name
            ancestry = [base_dir_name]  # Add base directory name at the beginning
            
            # Add intermediate directories
            ancestry.extend([part for part in rel_path.parts if part])
            
            # Add task name at the end
            ancestry.append(task_name)
            
            # Store ancestry information for this task
            task_ancestry[task_name] = ancestry
            
            logger.debug(f"Task '{task_name}' ancestry: {ancestry}")
    
    return task_ancestry

def find_all_task_configs(base_dir="tasks") -> tuple[dict[pathlib.Path, list[dict]], dict[str, list[str]]]:
    """
    Find all task configs in the base directory and its subdirectories.

    Args:
        base_dir: Base directory to search in, defaults to "tasks"

    Returns:
        A tuple containing:
        - Dictionary of task configs (mapping directory paths to lists of config dictionaries)
        - Dictionary of task ancestry (mapping task names to their folder hierarchy)
    """
    tasks_dir = pathlib.Path(base_dir)
    if not tasks_dir.exists():
        logger.warning("[find_all_task_configs] Tasks directory not found: %s", tasks_dir)
        return {}, {}
    
    groups = list(tasks_dir.glob("**/"))

    task_configs = {}
    for group in groups:
        task_configs[group] = get_all_configs_in_dir(group)

    if not task_configs:
        logger.warning("[find_all_task_configs] No task configs found in %s", tasks_dir)
        return {}, {}
    
    logger.info("[find_all_task_configs] Found %d task configs", len(task_configs[tasks_dir]))

    # Resolve all task configs
    for task_group, task_files in task_configs.items():
        task_configs[task_group] = [resolve_file(task_file) for task_file in task_files]
    
    # Build task ancestry information
    task_ancestry = build_task_ancestry(tasks_dir, task_configs)
    
    return task_configs, task_ancestry



def get_groups(task_configs):
    """
    Get all groups of tasks.
    
    Args:
        task_configs: List of task configs
    
    Returns:
        Dictionary of groups
    """
    return {x.name: x for x in task_configs.keys()}

def get_tasks(task_configs):
    """
    Get all tasks.
    
    Args:
        task_configs: List of task configs
    
    Returns:
        Dictionary of tasks
    """
    return {x['task_name']: x for x in list(task_configs.values())[0]}

def get_task_metric_pairs_from_config(cfg: dict) -> list[tuple[str, str]]:
    """
    Get task-metric pairs from config.
    
    Args:
        cfg: Configuration dictionary
    
    Returns:
        List of tuples (task_name, metric_name)
    """
    return cfg.get("task_metric", [])

def _validate_task_metric_pairs(task_metric_pairs, task_configs, task_ancestry=None):
    """ Validate task names and metrics in config.

    Args:
        task_metric_pairs: List of [task_name, metric_name] pairs from the config
        task_configs: List of task configs
    """

    # Get all groups of tasks. These are generally folders
    groups = get_groups(task_configs)

    # Get all tasks. These are generally YAML files
    tasks = get_tasks(task_configs)

    for task_metric_pair in task_metric_pairs:
        # Check if this is a valid pair format
        if not isinstance(task_metric_pair, list) or len(task_metric_pair) != 2:
            raise ValueError(f"Invalid task_metric pair format: {task_metric_pair}")

        task_name, metric_name = task_metric_pair
        
        # Validate task name and metric
        valid_metric = False
        if task_name in groups.keys():
            # Get all the supported metrics that are common to all tasks in that group
            all_valid_metrics = {metric['metric'] for task_config in task_configs[groups[task_name]] for metric in task_config.get('metrics', [])}
            all_valid_metrics.add("all")
            if metric_name in all_valid_metrics:
                valid_metric = True
        elif task_name in tasks.keys():
            # Get all the supported metrics for that task
            all_valid_metrics = {metric['metric'] for metric in tasks[task_name]['metrics']}
            all_valid_metrics.add("all")
            if metric_name in all_valid_metrics:
                valid_metric = True
        else:
            raise ValueError(f"Invalid task name: {task_name}")
        
        if not valid_metric:
            raise ValueError(f"Invalid metric name: {metric_name} for task {task_name}")

    return
    
def expand_task_metric_pairs(cfg: dict, task_configs: dict[pathlib.Path, list[dict]], task_ancestry: dict[str, list[str]] = None) -> list[tuple[str, str, dict, list[str]]]:
    """
    Expand task-metric pairs from config.
    
    Args:
        cfg: Configuration dictionary
        task_configs: List of task configs
        task_ancestry: Dictionary mapping task names to their ancestry paths (list of parent directories)
        
    Returns:
        List of tuples (task_name, metric_name, task_config, ancestry)
    """

    # Get all groups of tasks. These are generally folders
    groups = get_groups(task_configs)

    # Get all tasks. These are generally YAML files
    tasks = get_tasks(task_configs)

    task_metric_pairs = get_task_metric_pairs_from_config(cfg)

    task_payload = []

    for task_metric_pair in task_metric_pairs:
        task_name, metric_name = task_metric_pair

        # Using task ancestry if available
        if task_name in groups.keys():
            for task_config in task_configs[groups[task_name]]:
                config_task_name = task_config['task_name']
                ancestry = task_ancestry.get(config_task_name, []) if task_ancestry else []
                task_payload.append((config_task_name, metric_name, task_config, ancestry))
        elif task_name in tasks.keys():
            ancestry = task_ancestry.get(task_name, []) if task_ancestry else []
            task_payload.append((task_name, metric_name, tasks[task_name], ancestry))
        else:
            raise ValueError(f"Invalid task name: {task_name}")
    
    return task_payload