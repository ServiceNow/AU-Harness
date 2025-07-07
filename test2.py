from datasets import load_dataset_builder

# Use your token for authentication
builder = load_dataset_builder(
    "ServiceNow-AI/enterprise-audio",
    token="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"
)
print("Available splits:", builder.info.splits)