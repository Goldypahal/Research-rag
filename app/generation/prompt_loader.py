import yaml
import os

class PromptLoader:
    def __init__(self, prompts_dir="configs/prompts"):
        self.prompts_dir = prompts_dir

    def load_prompt(self, name: str, version: str = "v1") -> str:
        path = os.path.join(self.prompts_dir, f"{name}_{version}.yaml")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt {name} {version} not found.")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            return config['template']
