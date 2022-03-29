import os
import subprocess


def on_paths(func, *paths):
    for path in paths:
        try:
            if isinstance(path, (tuple, list)):
                on_paths(func, *path)
            else:
                func(path)
        except FileNotFoundError:
            pass


def assert_exists(*paths):
    def func(path):
        if not os.path.exists(path):
            dirpath = os.path.dirname(path)
            if os.path.exists(dirpath):
                msg = f"{path} is not in {dirpath}"
            else:
                msg = f"{dirpath} does not exist for {path}"
            assert False, msg

    on_paths(func, *paths)


def remove(*paths):
    on_paths(os.remove, *paths)


def content(path):
    with open(path) as f:
        return f.read()


def run(*cmd):
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(result.stderr)
    assert result.returncode == 0, " ".join(cmd)
    return str(result.stdout, encoding="utf-8")
