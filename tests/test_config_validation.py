import copy

import pytest
from pydantic import ValidationError

from src.config.validator import validate_config
from src.utils import load_config


def test_current_config_is_valid():
    assert validate_config(load_config("config/config.yaml"))


def test_unknown_config_field_is_rejected():
    cfg = copy.deepcopy(load_config("config/config.yaml"))
    cfg["vector_store"]["persist_direcotry"] = "typo"
    with pytest.raises(ValidationError, match="persist_direcotry"):
        validate_config(cfg)
