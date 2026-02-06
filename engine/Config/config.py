
## load yaml
import yaml

class Config:
    def __init__(self, config_file: str = 'detail_config/simple_MNIST_training.yaml'):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self):
        with open(self.config_file, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        return self.config
    
    def get(self, key: str, fallback = None):
        key_list = key.split('.')
        result = self.config
        for key in key_list:
            try:
                result = result[key]
            except KeyError:
                Warning("an error occurs when try get value at {}, fallback to {}", key, fallback)
                return fallback
        return result

def register_build_with_config(config_class: type[Config]):
    def decorator(func):
        setattr(config_class, func.__name__, func)
        return func
    return decorator

