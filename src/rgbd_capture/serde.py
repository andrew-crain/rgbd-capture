import json
import os
from typing import Type, TypeVar

import numpy as np
from pydantic import BaseModel
import yaml


T = TypeVar("T", bound=BaseModel)


def save_model(filename: str, model: BaseModel):
    base_name, file_extension = os.path.splitext(filename)

    with open(filename, "w") as f:
        if file_extension == ".yaml":
            yaml.dump(model.model_dump(mode="json"), f, default_flow_style=None)
        else:
            f.write(model.model_dump_json(indent=4))


def load_model(filename: str, model_type: Type[T]) -> T:
    base_name, file_extension = os.path.splitext(filename)

    with open(filename, "r") as f:
        if file_extension == ".yaml":
            return model_type.model_validate(yaml.safe_load(f))
        else:
            return model_type.model_validate(json.load(f))
