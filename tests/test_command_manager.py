from geon.tools.command_manager import CommandManager, LambdaCommand


def test_lambda_command_executes_and_undos() -> None:
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
