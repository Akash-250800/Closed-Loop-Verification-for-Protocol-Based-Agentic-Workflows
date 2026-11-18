import asyncio
from types import SimpleNamespace

import pytest

from benchmarks import multiref_tasks
from multiref_server import MultiRefData
from verification import validate, unwrap_mcp_reply


def test_multiref_tasks_are_chained_and_do_not_expose_expected_tool():
    tasks = multiref_tasks()
    assert len(tasks) >= 8
    assert all(len(task["steps"]) == 3 for task in tasks)
    for task in tasks:
        assert all("expected_tool" not in step for step in task["steps"])
        assert task["steps"][1]["grounding"]["history_field"] == "country_code"
        assert task["steps"][2]["grounding"]["history_field"] == "presidency_reference"


def test_pure_three_reference_chain():
    data = MultiRefData()
    country = data.country_by_capital("Paris")
    assert country == {"capital": "Paris", "country_code": "FRA", "country_name": "France"}
    index = data.president_ids(country["country_code"])
    assert index["presidency_reference"] == "FRA_PRESIDENTS"
    assert index["president_ids"] == ["FR_JC", "FR_NS", "FR_FH"]
    term = data.president_by_date(index["presidency_reference"], "2010-07-01")
    assert term["president_name"] == "Nicolas Sarkozy"


def test_all_benchmark_gold_outputs_match_three_reference_data():
    data = MultiRefData()
    for task in multiref_tasks():
        first = task["steps"][0]
        second = task["steps"][1]
        third = task["steps"][2]
        capital = first["diagnostic_call"]["args"]["capital"]
        country = data.country_by_capital(capital)
        assert country == first["expected_output"]
        index = data.president_ids(country["country_code"])
        assert index == second["expected_output"]
        target = third["diagnostic_call"]["args"]["target_date"]
        term = data.president_by_date(index["presidency_reference"], target)
        assert term == third["expected_output"]


def test_grounding_rejects_hallucinated_country_code():
    catalog = {
        "lookup_president_ids": SimpleNamespace(
            input_schema={
                "type": "object",
                "properties": {"country_code": {"type": "string"}},
                "required": ["country_code"],
                "additionalProperties": False,
            }
        )
    }
    step = multiref_tasks()[0]["steps"][1]
    history = [{"output": {"capital": "Paris", "country_code": "FRA", "country_name": "France"}}]
    err = validate(
        {"tool": "lookup_president_ids", "args": {"country_code": "USA"}},
        catalog,
        step,
        history=history,
    )
    assert err is not None
    assert err.code == "ungrounded_argument"


def test_type_mismatch_feedback_is_grounded_in_previous_mcp_value():
    catalog = {
        "lookup_president_ids": SimpleNamespace(
            input_schema={
                "type": "object",
                "properties": {"country_code": {"type": "string"}},
                "required": ["country_code"],
                "additionalProperties": False,
            }
        )
    }
    step = multiref_tasks()[0]["steps"][1]
    history = [{"output": {"capital": "Washington D.C.", "country_code": "USA", "country_name": "United States"}}]
    err = validate(
        {"tool": "lookup_president_ids", "args": {"country_code": 12345}},
        catalog,
        step,
        history=history,
    )
    assert err is not None
    assert err.code == "jsonschema_type"
    assert "USA" in err.message
    assert "USA" in err.expected


def test_wrong_tool_feedback_can_infer_correct_tool_from_grounded_argument():
    catalog = {
        "lookup_country_by_capital": SimpleNamespace(
            input_schema={
                "type": "object",
                "properties": {"capital": {"type": "string"}},
                "required": ["capital"],
                "additionalProperties": False,
            }
        ),
        "lookup_president_ids": SimpleNamespace(
            input_schema={
                "type": "object",
                "properties": {"country_code": {"type": "string"}},
                "required": ["country_code"],
                "additionalProperties": False,
            }
        ),
    }
    step = multiref_tasks()[0]["steps"][1]
    history = [{"output": {"capital": "Washington D.C.", "country_code": "USA", "country_name": "United States"}}]
    # This mirrors the injected wrong-tool shape: args still contain country_code.
    err = validate(
        {"tool": "lookup_country_by_capital", "args": {"country_code": "USA"}},
        catalog,
        step,
        history=history,
    )
    assert err is not None
    assert err.suggested_tool == "lookup_president_ids"
    assert "USA" in err.message


def test_grounding_rejects_hallucinated_presidency_reference():
    catalog = {
        "lookup_president_by_date": SimpleNamespace(
            input_schema={
                "type": "object",
                "properties": {
                    "presidency_reference": {"type": "string"},
                    "target_date": {"type": "string"},
                },
                "required": ["presidency_reference", "target_date"],
                "additionalProperties": False,
            }
        )
    }
    step = multiref_tasks()[0]["steps"][2]
    history = [{"output": {
        "country_code": "USA",
        "presidency_reference": "USA_PRESIDENTS",
        "president_ids": ["US_GWB", "US_BO", "US_DT1"],
    }}]
    err = validate(
        {"tool": "lookup_president_by_date", "args": {
            "presidency_reference": "FRA_PRESIDENTS", "target_date": "2012-07-01"
        }},
        catalog,
        step,
        history=history,
    )
    assert err is not None
    assert err.code == "ungrounded_argument"


def test_multiref_server_reference_chain():
    pytest.importorskip("mcp")
    from multiref_server import make_multiref_server
    from mcp.client import Client

    async def run():
        async with Client(make_multiref_server()) as client:
            a = await client.call_tool("lookup_country_by_capital", {"capital": "Paris"})
            country, err = unwrap_mcp_reply(a)
            assert err is None
            assert country["country_code"] == "FRA"

            b = await client.call_tool("lookup_president_ids", {"country_code": "FRA"})
            index, err = unwrap_mcp_reply(b)
            assert err is None
            assert index["presidency_reference"] == "FRA_PRESIDENTS"
            assert index["president_ids"] == ["FR_JC", "FR_NS", "FR_FH"]

            c = await client.call_tool(
                "lookup_president_by_date",
                {"presidency_reference": index["presidency_reference"], "target_date": "2010-07-01"},
            )
            term, err = unwrap_mcp_reply(c)
            assert err is None
            assert term["president_name"] == "Nicolas Sarkozy"
    asyncio.run(run())
