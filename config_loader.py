import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def get_config_section(config: Dict[str, Any], section: str) -> Dict[str, Any]:
    """Get a specific section from the configuration."""
    if section not in config:
        raise KeyError(f"Configuration section '{section}' not found")
    
    return config[section]
