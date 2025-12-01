import json
from dataclasses import asdict, dataclass


@dataclass
class Individual:
    stdout_filepath: str | None = None
    code_path: str | None = None
    code: str | None = None
    response_id: int | None = None
    exec_success: bool | None = None
    obj: float | None = None
    traceback_msg: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "Individual":
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "Individual":
        return cls.from_dict(json.loads(json_str))
