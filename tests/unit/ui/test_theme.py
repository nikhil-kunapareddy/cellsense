"""Pins cellsense.ui.theme: mode resolution, NO_COLOR precedence, token
coverage across every palette, and make_console wiring.
"""

from __future__ import annotations

import pytest

from cellsense.ui.theme import TOKENS, Theme, make_console, resolve_theme


def test_dark_mode_resolves_with_expected_mode_label(monkeypatch: pytest.MonkeyPatch) -> None:
    # NO_COLOR outranks an explicit mode by design, so it must be cleared here or
    # this test passes or fails depending on the developer's shell.
    monkeypatch.delenv("NO_COLOR", raising=False)
    theme = resolve_theme("dark")
    assert theme.mode == "dark"


def test_light_mode_resolves_with_expected_mode_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    theme = resolve_theme("light")
    assert theme.mode == "light"


def test_no_color_mode_maps_every_token_to_empty_string() -> None:
    theme = resolve_theme("no-color")
    assert theme.mode == "no-color"
    assert theme.styles == dict.fromkeys(TOKENS, "")


def test_no_color_env_var_forces_no_color_regardless_of_requested_mode(monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    theme = resolve_theme("dark")
    assert theme.mode == "no-color"


def test_no_color_env_var_wins_even_for_light_mode(monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "anything")
    theme = resolve_theme("light")
    assert theme.mode == "no-color"


def test_auto_mode_defaults_to_dark_without_colorfgbg(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("COLORFGBG", raising=False)
    assert resolve_theme("auto").mode == "dark"


def test_auto_mode_detects_light_background_from_colorfgbg(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("COLORFGBG", "0;15")
    assert resolve_theme("auto").mode == "light"


def test_auto_mode_detects_dark_background_from_colorfgbg(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("COLORFGBG", "15;0")
    assert resolve_theme("auto").mode == "dark"


def test_every_token_has_a_defined_style_in_dark_and_light_palettes(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    dark = resolve_theme("dark")
    light = resolve_theme("light")
    assert set(TOKENS) <= set(dark.styles)
    assert set(TOKENS) <= set(light.styles)


def test_unknown_token_degrades_to_empty_style_instead_of_raising() -> None:
    theme = Theme(mode="dark", styles={"known": "bold"})
    assert theme.style("totally.unknown.token") == ""


def test_theme_is_frozen() -> None:
    theme = resolve_theme("dark")
    try:
        theme.mode = "light"  # type: ignore[misc]
        raised = False
    except Exception:
        raised = True
    assert raised


def test_make_console_no_color_flag_follows_theme_mode(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    no_color_console = make_console(resolve_theme("no-color"))
    dark_console = make_console(resolve_theme("dark"))
    assert no_color_console.no_color is True
    assert dark_console.no_color is False


def test_make_console_disables_markup_highlighting() -> None:
    console = make_console(resolve_theme("dark"))
    assert console._highlight is False
