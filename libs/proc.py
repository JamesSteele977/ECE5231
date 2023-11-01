import json, os

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
    with open(path, 'r') as f:
        test = f.read()
        if (test is None) or (test == '{}'):
            data = {}
        else:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
                print(f"Error encountered in {path} read; returning empty")
    return data

def write_json_dict(path: str, data: dict):
    _check_(path, 'json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    pass

def read_md(path: str) -> str:
    _check_(path, 'md')
    with open(path, 'r') as f:
        data = f.read()
    return data