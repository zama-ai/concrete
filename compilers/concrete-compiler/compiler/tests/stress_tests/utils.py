import subprocess

CONCRETECOMPILER = 'concretecompiler'

def ceil_log2(v, exact=False):
    import math
    log_v = math.ceil(math.log(v) / math.log(2))
    assert not exact or v == 2 ** log_v
    return log_v

def log2(v: int):
    return ceil_log2(v, exact=True)

def run(*cmd):
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(result.stderr)
    assert result.returncode == 0, ' '.join(cmd)
    return str(result.stdout, encoding='utf-8')
