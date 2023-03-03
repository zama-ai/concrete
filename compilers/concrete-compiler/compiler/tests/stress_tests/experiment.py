from dataclasses import asdict, dataclass
import json

@dataclass
class Replication:
    success: bool
    details: 'list[str]'

@dataclass
class ExperimentConditions:
    bitwidth: int
    size: int
    args: 'list[int]'
    log_manp_max: int
    overflow: bool
    details: 'list[str]'

@dataclass
class Experiment:
    cmd: str
    conditions: ExperimentConditions
    replications: 'list[Replication]'
    code: str
    success_rate: float
    overflow_rate: float

class Encoder(json.JSONEncoder):
    def default(self, z):
        try:
            return super().default(z)
        except:
            return asdict(z)
