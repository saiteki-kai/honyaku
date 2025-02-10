def model_name_to_path(model_name: str) -> str:
    """Converts a model name to a path-safe string."""
    return model_name.replace("/", "__")
