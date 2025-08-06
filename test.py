import yaml

def convert_to_json_style_lists(yaml_file_path="dataset_metric_suggestions.yaml"):
    """
    Reads a YAML file, converts stringified tuples to JSON-style lists of strings,
    and writes the converted data to the YAML file.
    """
    with open(yaml_file_path, 'r') as file:
        content = file.read()
        data = yaml.safe_load(content)

    converted_data = {"dataset_metric": []}
    
    if "dataset_metric" in data:
        for item in data["dataset_metric"]:
            # If it's already a list (from previous conversion)
            if isinstance(item, list):
                # Format as ["dataset", "metric"]
                converted_data["dataset_metric"].append([item[0], item[1]])
            else:
                # Remove parentheses and split by comma
                cleaned_item = item.strip('()')
                parts = [part.strip() for part in cleaned_item.split(',')]
                converted_data["dataset_metric"].append([parts[0], parts[1]])
    
    # Convert to YAML format with JSON-style lists
    yaml_output = "dataset_metric:\n"
    for pair in converted_data["dataset_metric"]:
        yaml_output += f'- ["{pair[0]}", "{pair[1]}"]\n'
    
    with open(yaml_file_path, 'w') as file:
        file.write(yaml_output)

if __name__ == "__main__":
    convert_to_json_style_lists()
    print("Conversion complete!")