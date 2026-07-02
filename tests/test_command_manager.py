from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_command_manager_module():
    path = Path(__file__).resolve().parent.parent / "src" / "geon" / "tools" / "command_manager.py"
    spec = importlib.util.spec_from_file_location("_geon_command_manager_for_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lambda_command_executes_and_undos() -> None:
    command_manager = _load_command_manager_module()
    CommandManager = command_manager.CommandManager
    LambdaCommand = command_manager.LambdaCommand
    events: list[str] = []
    manager = CommandManager()

    manager.do(
        LambdaCommand(
            "record",
            lambda: events.append("execute"),
            lambda: events.append("undo"),
        )
    )
    manager.undo()

    assert events == ["execute", "undo"]
