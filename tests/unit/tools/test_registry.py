"""Pins cellsense.tools.registry.ToolRegistry: discovery/uniqueness, the
run() execution path, and every schema shape (OpenAI, Anthropic, LangChain).
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from cellsense.errors import ToolError
from cellsense.io.schema import ToolResult, Workspace
from cellsense.tools.base import tool_spec
from cellsense.tools.registry import ToolRegistry

EXPECTED_TOOL_NAMES = {
    "aggregate",
    "describe",
    "filter_rows",
    "join",
    "plot",
    "list_directory",
    "find_files",
}


def _valid_json_schema(schema: dict) -> None:
    """A hand-rolled, dependency-free structural check (no `jsonschema`
    package is in this project's dev extras) -- not a full draft-07
    validator, but enough to catch a malformed schema (bad "type", "required"
    naming a property that doesn't exist, non-dict "properties" entries...).
    """
    valid_types = {"object", "array", "string", "integer", "number", "boolean", "null"}
    assert isinstance(schema, dict)
    assert schema.get("type") in valid_types
    properties = schema.get("properties", {})
    assert isinstance(properties, dict)
    for name, prop_schema in properties.items():
        assert isinstance(name, str)
        assert isinstance(prop_schema, dict)
        if "type" in prop_schema:
            assert prop_schema["type"] in valid_types
        if prop_schema.get("type") == "array" and "items" in prop_schema:
            assert isinstance(prop_schema["items"], dict)
        if "enum" in prop_schema:
            assert isinstance(prop_schema["enum"], list)
    required = schema.get("required", [])
    assert isinstance(required, list)
    for key in required:
        assert key in properties, f"required key {key!r} is not a declared property"


class TestDefaultRegistry:
    def test_builtin_tool_set_matches_the_seven_documented_tools(self) -> None:
        registry = ToolRegistry()
        assert {s.name for s in registry.specs()} == EXPECTED_TOOL_NAMES

    def test_names_are_unique(self) -> None:
        registry = ToolRegistry()
        names = [s.name for s in registry.specs()]
        assert len(names) == len(set(names))

    def test_get_returns_the_matching_spec(self) -> None:
        registry = ToolRegistry()
        assert registry.get("aggregate").name == "aggregate"

    def test_get_unknown_tool_raises_tool_error_listing_available(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolError) as exc_info:
            registry.get("nope")
        assert "aggregate" in (exc_info.value.hint or "")

    def test_run_executes_the_handler_and_returns_a_tool_result(
        self, sales_df, make_workspace
    ) -> None:
        registry = ToolRegistry()
        ws = make_workspace(sales=sales_df)
        result = registry.run("describe", {"columns": ["region"]}, ws)
        assert isinstance(result, ToolResult)

    def test_run_validates_args_before_calling_the_handler(self, make_workspace, sales_df) -> None:
        registry = ToolRegistry()
        ws = make_workspace(sales=sales_df)
        with pytest.raises(ToolError, match="missing required argument"):
            registry.run("filter_rows", {}, ws)

    def test_run_coerces_stringly_typed_booleans_before_calling_the_handler(
        self, make_workspace, sales_df
    ) -> None:
        registry = ToolRegistry()
        ws = make_workspace(sales=sales_df)
        # find_files' "recursive" is boolean; registry.run must coerce "false"
        # the same way a raw handler call via ToolSpec.validate_args would.
        result = registry.run("find_files", {"recursive": "false"}, ws)
        assert isinstance(result, ToolResult)


class TestDuplicateNameDetection:
    def test_constructing_with_duplicate_names_raises(self) -> None:
        def _handler(args: dict[str, Any], workspace: Workspace) -> ToolResult:
            return ToolResult(data=None, summary="", citations=[])  # type: ignore[arg-type]

        spec_a = tool_spec(name="dup", description="a", parameters={"type": "object"})(_handler)
        spec_b = tool_spec(name="dup", description="b", parameters={"type": "object"})(_handler)
        with pytest.raises(ValueError, match="Duplicate tool name"):
            ToolRegistry(specs=[spec_a, spec_b])

    def test_custom_restricted_registry_only_exposes_given_specs(self) -> None:
        def _handler(args: dict[str, Any], workspace: Workspace) -> ToolResult:
            return ToolResult(data=None, summary="", citations=[])  # type: ignore[arg-type]

        spec = tool_spec(name="solo", description="only tool", parameters={"type": "object"})(
            _handler
        )
        registry = ToolRegistry(specs=[spec])
        assert {s.name for s in registry.specs()} == {"solo"}


class TestSchemaExports:
    def test_openai_schemas_shape(self) -> None:
        registry = ToolRegistry()
        for schema in registry.openai_schemas():
            assert schema["type"] == "function"
            assert set(schema["function"]) == {"name", "description", "parameters"}
            assert schema["function"]["name"] in EXPECTED_TOOL_NAMES
            _valid_json_schema(schema["function"]["parameters"])

    def test_anthropic_schemas_shape(self) -> None:
        registry = ToolRegistry()
        for schema in registry.anthropic_schemas():
            assert set(schema) == {"name", "description", "input_schema"}
            assert schema["name"] in EXPECTED_TOOL_NAMES
            _valid_json_schema(schema["input_schema"])

    def test_every_tool_parameters_schema_is_valid(self) -> None:
        registry = ToolRegistry()
        for spec in registry.specs():
            _valid_json_schema(spec.parameters)

    def test_langchain_tools_are_bindable(self) -> None:
        registry = ToolRegistry()
        tools = registry.langchain_tools()
        assert len(tools) == len(EXPECTED_TOOL_NAMES)
        for tool in tools:
            assert tool.name in EXPECTED_TOOL_NAMES
            assert tool.description
            assert issubclass(tool.args_schema, BaseModel)
            # This is exactly what model.bind_tools() does under the hood
            # (via convert_to_openai_tool) -- it must not raise.
            json_schema = tool.args_schema.model_json_schema()
            assert isinstance(json_schema, dict)

    def test_langchain_tool_is_never_actually_invoked(self) -> None:
        registry = ToolRegistry()
        tool = next(t for t in registry.langchain_tools() if t.name == "describe")
        with pytest.raises(ToolError, match="never executes directly"):
            tool.func()
