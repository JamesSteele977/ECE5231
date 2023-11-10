import json, os

def deref_iter(obj):
    if hasattr(obj, '__iter__'):
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            return dict(obj)
        elif isinstance(obj, tuple) or isinstance(obj, set) or isinstance(obj, list):
            return tuple(obj)
        else:
            try:
                return tuple(obj)
            except TypeError:
                pass
    return obj

def _check_path(path: str):
    if type(path) is not str:
        raise TypeError(f"Invalid argument type: {type(path)}")
    if not os.path.exists(path):
        raise ValueError(f"Error: path does not exist: {path}")
    pass

def _check_(path: str, ftype: str):
    _check_path(path)
    if path.split('.')[-1] != ftype:
        raise ValueError(f"File {path} is not json")
    pass

def read_json(path: str) -> dict:
    _check_(path, 'json')
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        print(f"Incorrect formatting during read file {path}. Overwriting.")
        with open(path, 'w') as f:
            json.dump({}, f)
        return {}

def write_json(path: str, data: dict):
    _check_(path, 'json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    pass

def read_md(path: str) -> str:
    _check_(path, 'md')
    with open(path, 'r') as f:
        data = f.read()
    return data