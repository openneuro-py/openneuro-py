"""Tests for the print-or-log switch in `openneuro._console`."""

import logging

import pytest
from typer.testing import CliRunner

import openneuro._cli
import openneuro._console
from openneuro._console import cprint


@pytest.fixture
def cli_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend we were invoked through the command line interface."""
    monkeypatch.setattr(openneuro._console, "_RUNNING_FROM_CLI", True)


def test_library_mode_logs(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    """Without the CLI flag, messages become records on the openneuro logger."""
    cprint("hello from the library")

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.name == "openneuro"
    assert record.levelno == logging.INFO
    assert record.getMessage() == "hello from the library"

    captured = capsys.readouterr()
    assert captured.out == ""
    # The default handler still renders them, through the shared console.
    assert captured.err.count("hello from the library") == 1


def test_library_mode_splits_lines(caplog: pytest.LogCaptureFixture) -> None:
    """Multi-line messages become one record per line, blank lines dropped."""
    cprint("\nfirst\n\nsecond\n")

    assert [r.getMessage() for r in caplog.records] == ["first", "second"]


def test_library_mode_honors_level(caplog: pytest.LogCaptureFixture) -> None:
    """Failures and retries can be logged above INFO."""
    cprint("something went wrong", level=logging.WARNING)

    assert [r.levelno for r in caplog.records] == [logging.WARNING]


def test_library_mode_can_be_silenced(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    """Raising the level of the openneuro logger silences the messages."""
    logger = logging.getLogger("openneuro")
    original = logger.level
    try:
        logger.setLevel(logging.WARNING)
        cprint("not interesting")
    finally:
        logger.setLevel(original)

    assert caplog.records == []
    assert capsys.readouterr().err == ""


@pytest.mark.usefixtures("cli_mode")
def test_cli_mode_prints(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    """With the CLI flag, messages are printed and never logged."""
    cprint("hello from the CLI")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "hello from the CLI" in captured.err
    assert caplog.records == []


def test_cli_only_is_dropped_in_library_mode(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    """Decorative messages are not worth a log record."""
    cprint("please enjoy your brains", cli_only=True)

    assert caplog.records == []
    assert capsys.readouterr().err == ""


@pytest.mark.usefixtures("cli_mode")
def test_cli_only_is_shown_in_cli_mode(capsys: pytest.CaptureFixture[str]) -> None:
    """Decorative messages are exactly what the CLI is for."""
    cprint("please enjoy your brains", cli_only=True)

    assert "please enjoy your brains" in capsys.readouterr().err


def test_cli_sets_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The app callback flips the flag before any subcommand runs."""
    seen = []

    def fake_download(**kwargs: object) -> None:
        seen.append(openneuro._console._RUNNING_FROM_CLI)

    monkeypatch.setattr(openneuro._cli, "download", fake_download)
    result = CliRunner().invoke(openneuro._cli.app, ["download", "--dataset=ds000248"])

    assert result.exit_code == 0, result.output
    assert seen == [True]
