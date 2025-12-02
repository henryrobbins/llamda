import json
import logging
from dataclasses import asdict, dataclass


logger = logging.getLogger("llamda")


@dataclass
class Individual:
    name: str | None = None
    code: str | None = None
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

    def write_code_to_file(self, filepath: str) -> None:
        if self.code is None:
            logger.warning("No code to write to file.")
            return
        with open(filepath, "w") as f:
            f.write(self.code)
