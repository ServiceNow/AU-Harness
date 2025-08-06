import yaml

def convert_dataset_metrics(yaml_file_path="dataset_metric_suggestions.yaml"):
    """
    Reads a YAML file, converts stringified tuples to lists of strings,
    and returns the converted data.
    """
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)

    converted_data = []
    if "dataset_metric" in data:
        for item in data["dataset_metric"]:
            # Remove parentheses and split by comma
            cleaned_item = item.strip('()')
            parts = [part.strip() for part in cleaned_item.split(',')]
            converted_data.append(parts)
    return converted_data

if __name__ == "__main__":
    converted_list = convert_dataset_metrics()
    for item in converted_list:
        print(item)
