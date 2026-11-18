"""Built-in multi-tool sandbox and official BFCL JSONL adapter (where compatible)."""
import json
import itertools
from pathlib import Path
from inventory_data import INITIAL_STOCK, RESERVE_AMOUNTS


def inventory_tasks():
    result = []
    for sku, qty in RESERVE_AMOUNTS.items():
        before = INITIAL_STOCK[sku]
        result.append({
            "id": f"inventory_{sku.lower()}", "source": "synthetic_inventory",
            "description": f"Check the stock for {sku}, reserve exactly {qty} units, and then confirm the new stock level.",
            "fault_step": 1,
            "steps": [
                {"instruction": f"Check current stock of {sku}.", "expected_tool":"check_stock", "expected_args":{"sku":sku},
                 "expected_output":{"sku":sku,"available":before},"mutating":False},
                {"instruction": f"Reserve exactly {qty} units of SKU {sku}.","expected_tool":"reserve_stock","expected_args":{"sku":sku,"quantity":qty},
                 "expected_output":{"sku":sku,"reserved":qty,"remaining":before-qty},"mutating":True},
                {"instruction": f"Check remaining stock for {sku} after reservation.","expected_tool":"check_stock","expected_args":{"sku":sku},
                 "expected_output":{"sku":sku,"available":before-qty},"mutating":False},
            ],
        })
    return result



def multiref_tasks():
    """Genuine three-reference tasks: capital -> country -> presidency reference -> term.

    Later calls require identifiers returned by earlier MCP calls. The real LLM
    is not given an expected tool. The second reference also returns the list of
    president IDs for auditability, while a string presidency_reference is used
    as the dependent key for the third call. Using a string key avoids making
    the experiment depend on a small model's array-serialization quirks.
    """
    cases = [
        # First four are one task per country so a short run still demonstrates source diversity.
        ("washington_2012", "Washington D.C.", "2012-07-01", "USA", "United States", "USA_PRESIDENTS",
         ["US_GWB", "US_BO", "US_DT1"], "US_BO", "Barack Obama", "2009-01-20", "2017-01-20"),
        ("paris_2010", "Paris", "2010-07-01", "FRA", "France", "FRA_PRESIDENTS",
         ["FR_JC", "FR_NS", "FR_FH"], "FR_NS", "Nicolas Sarkozy", "2007-05-16", "2012-05-15"),
        ("mexico_2010", "Mexico City", "2010-07-01", "MEX", "Mexico", "MEX_PRESIDENTS",
         ["MX_VF", "MX_FC", "MX_EPN"], "MX_FC", "Felipe Calderon", "2006-12-01", "2012-12-01"),
        ("brasilia_2014", "Brasilia", "2014-07-01", "BRA", "Brazil", "BRA_PRESIDENTS",
         ["BR_LULA1", "BR_DR", "BR_MT"], "BR_DR", "Dilma Rousseff", "2011-01-01", "2016-08-31"),
        ("washington_2004", "Washington D.C.", "2004-07-01", "USA", "United States", "USA_PRESIDENTS",
         ["US_GWB", "US_BO", "US_DT1"], "US_GWB", "George W. Bush", "2001-01-20", "2009-01-20"),
        ("paris_2004", "Paris", "2004-07-01", "FRA", "France", "FRA_PRESIDENTS",
         ["FR_JC", "FR_NS", "FR_FH"], "FR_JC", "Jacques Chirac", "1995-05-17", "2007-05-16"),
        ("mexico_2004", "Mexico City", "2004-07-01", "MEX", "Mexico", "MEX_PRESIDENTS",
         ["MX_VF", "MX_FC", "MX_EPN"], "MX_VF", "Vicente Fox", "2000-12-01", "2006-12-01"),
        ("brasilia_2006", "Brasilia", "2006-07-01", "BRA", "Brazil", "BRA_PRESIDENTS",
         ["BR_LULA1", "BR_DR", "BR_MT"], "BR_LULA1", "Luiz Inacio Lula da Silva", "2003-01-01", "2011-01-01"),
        ("washington_2018", "Washington D.C.", "2018-07-01", "USA", "United States", "USA_PRESIDENTS",
         ["US_GWB", "US_BO", "US_DT1"], "US_DT1", "Donald Trump", "2017-01-20", "2021-01-20"),
        ("paris_2014", "Paris", "2014-07-01", "FRA", "France", "FRA_PRESIDENTS",
         ["FR_JC", "FR_NS", "FR_FH"], "FR_FH", "Francois Hollande", "2012-05-15", "2017-05-14"),
        ("mexico_2016", "Mexico City", "2016-07-01", "MEX", "Mexico", "MEX_PRESIDENTS",
         ["MX_VF", "MX_FC", "MX_EPN"], "MX_EPN", "Enrique Pena Nieto", "2012-12-01", "2018-12-01"),
        ("brasilia_2018", "Brasilia", "2018-07-01", "BRA", "Brazil", "BRA_PRESIDENTS",
         ["BR_LULA1", "BR_DR", "BR_MT"], "BR_MT", "Michel Temer", "2016-08-31", "2019-01-01"),
    ]
    tasks=[]
    for ident, capital, target, code, country, president_reference, ids, president_id, president_name, start, end in cases:
        tasks.append({
            "id": f"multiref_{ident}",
            "source": "multiref_presidents",
            "description": (
                f"Who was president of the country whose capital is {capital} on {target}? "
                "Answer only after resolving the country and presidential term through the MCP references."
            ),
            "fault_step": 1,
            "steps": [
                {
                    "instruction": (
                        f"Resolve the country whose capital is {capital} using the available MCP reference. "
                        "Do not answer from memory."
                    ),
                    "diagnostic_call": {"tool":"lookup_country_by_capital","args":{"capital":capital}},
                    "expected_output": {"capital":capital,"country_code":code,"country_name":country},
                    "mutating": False,
                },
                {
                    "instruction": (
                        "Using the country_code returned by the previous successful MCP call, retrieve the "
                        "presidency reference and president identifiers for that country. Do not invent or substitute a country code."
                    ),
                    "diagnostic_call": {"tool":"lookup_president_ids","args":{"country_code":code}},
                    "grounding": {"argument":"country_code","history_field":"country_code","mode":"equals"},
                    "expected_output": {
                        "country_code":code,
                        "presidency_reference":president_reference,
                        "president_ids":ids,
                    },
                    "mutating": False,
                },
                {
                    "instruction": (
                        f"Using the presidency_reference returned by the previous successful MCP call, find which "
                        f"presidential term contains {target}. Use that retrieved reference exactly; do not invent a reference."
                    ),
                    "diagnostic_call": {
                        "tool":"lookup_president_by_date",
                        "args":{"presidency_reference":president_reference,"target_date":target},
                    },
                    "grounding": {"argument":"presidency_reference","history_field":"presidency_reference","mode":"equals"},
                    "expected_output": {
                        "president_id":president_id,"president_name":president_name,"target_date":target,
                        "term_start":start,"term_end":end,
                    },
                    "mutating": False,
                },
            ],
        })
    return tasks

def _extract_user_text(question):
    if isinstance(question, str):
        return question
    fragments = []
    def recurse(obj):
        if isinstance(obj, str):
            fragments.append(obj)
        elif isinstance(obj, dict):
            if obj.get("role") == "user" and isinstance(obj.get("content"), str):
                fragments.append(obj["content"])
            elif "content" in obj:
                recurse(obj["content"])
            elif "question" in obj:
                recurse(obj["question"])
        elif isinstance(obj, list):
            for element in obj:
                recurse(element)
    recurse(question)
    return "\n".join(fragments)


def _gold_variants(record):
    """Supported BFCL single-call: {function_name: {arg:[possible_values]}}."""
    entries = record.get("ground_truth") or record.get("answer") or []
    if isinstance(entries, dict):
        entries = [entries]
    parsed = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for name, params in entry.items():
            if not isinstance(params, dict):
                continue
            keys = list(params)
            values = [v if isinstance(v, list) else [v] for v in params.values()]
            for combination in itertools.product(*values):
                parsed.append({"tool": name, "args": dict(zip(keys,combination))})
                if len(parsed) > 128:
                    raise ValueError("More than 128 valid gold combinations; exclude this case")
    return parsed


def _normalize_bfcl_schema(schema):
    """BFCL uses `type: dict`, and sometimes `float`; JSON Schema uses object/number."""
    if isinstance(schema, list):
        return [_normalize_bfcl_schema(x) for x in schema]
    if isinstance(schema, dict):
        output = {k: _normalize_bfcl_schema(v) for k,v in schema.items()}
        if output.get("type") == "dict":
            output["type"] = "object"
        elif output.get("type") == "float":
            output["type"] = "number"
        elif output.get("type") == "tuple":
            output["type"] = "array"
        return output
    return schema


def load_bfcl(question_path, answer_path, limit=None):
    """Adapt **single-call JSONL** question/function + possible_answer files from BFCL.

    Unsupported parallel/multi-turn/AST-expression cases are rejected rather than
    silently guessed. This does not claim to reproduce official BFCL scoring.
    """
    if not question_path or not answer_path:
        raise ValueError("--bfcl-questions and --bfcl-answers are required for BFCL tasks")
    with Path(answer_path).open(encoding="utf-8") as fh:
        gold = {str(x["id"]): x for line in fh if (x := json.loads(line.strip()))}
    tasks, rejected = [], []
    with Path(question_path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            identifier = str(record.get("id", ""))
            tools = record.get("function")
            # Guard against falsely treating BFCL multi-turn or parallel cases as single-call.
            if "multi_turn" in identifier or "parallel" in identifier:
                rejected.append({"id":identifier,"reason":"parallel/multi-turn cases need their official evaluator"})
                continue
            question_data=record.get("question", "")
            user_turns=[]
            def collect_user_turns(item):
                if isinstance(item,dict):
                    if item.get("role")=="user":
                        user_turns.append(item)
                    for value in item.values():
                        if isinstance(value,(list,dict)):
                            collect_user_turns(value)
                elif isinstance(item,list):
                    for value in item:
                        collect_user_turns(value)
            collect_user_turns(question_data)
            if len(user_turns)>1:
                rejected.append({"id":identifier,"reason":"multiple user turns unsupported by single-call adapter"})
                continue
            choices = _gold_variants(gold.get(identifier, {}))
            prompt = _extract_user_text(record.get("question", ""))
            if not prompt or not isinstance(tools, list) or not tools or not choices:
                rejected.append({"id":identifier,"reason":"unsupported question/functions/ground_truth"})
                continue
            normalized = []
            for t in tools:
                try:
                    name = t["name"]
                    schema = t.get("parameters", {"type":"object","properties":{}})
                    if isinstance(schema, str):
                        schema = json.loads(schema)
                    if isinstance(schema.get("properties"), str):
                        schema["properties"] = json.loads(schema["properties"])
                    schema = _normalize_bfcl_schema(schema)
                    if schema.get("type") != "object":
                        raise ValueError("non-object schema")
                    normalized.append({"name":name, "description":t.get("description", ""),"input_schema":schema})
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    normalized=[]
                    break
            if not normalized:
                rejected.append({"id":identifier,"reason":"unsupported tool schema"})
                continue
            tasks.append({"id":identifier,"source":"bfcl_single_call", "description":prompt,"bfcl_tools":normalized,
                          "fault_step":0, "accepted_calls":choices,
                          "steps":[{"instruction":prompt,"expected_tool":"", "expected_args":{},
                                    "mutating":False,"bfcl_gold":choices}]})
            if limit and len(tasks) >= limit:
                break
    if not tasks:
        raise RuntimeError(f"No compatible BFCL tasks could be adapted; rejected {len(rejected)} entries. Try a BFCL simple/multiple single-call JSONL subset.")
    return tasks, rejected


def make_bfcl_server(task):
    """Low-level MCP server preserves each benchmark tool's original JSON Schema."""
    from mcp.server import Server
    from mcp.types import Tool, ListToolsResult, CallToolResult, TextContent
    from jsonschema import Draft202012Validator
    tools = [Tool(name=x["name"],description=x["description"],input_schema=x["input_schema"]) for x in task["bfcl_tools"]]
    catalog = {t.name: t for t in tools}
    async def on_list_tools(ctx, params):
        return ListToolsResult(tools=tools)
    async def on_call_tool(ctx, params):
        if params.name not in catalog:
            return CallToolResult(content=[TextContent(type="text",text="Unknown tool")],is_error=True)
        args = params.arguments or {}
        errors = list(Draft202012Validator(catalog[params.name].input_schema).iter_errors(args))
        if errors:
            return CallToolResult(content=[TextContent(type="text",text=f"Schema: {errors[0].message}")],is_error=True)
        # No external BFCL side effects are executed: a sandbox call and AST/oracle check.
        data={"tool":params.name,"args":args}
        return CallToolResult(content=[TextContent(type="text",text=json.dumps(data))],structured_content=data)
    return Server("BFCL MCP schema adapter", on_list_tools=on_list_tools, on_call_tool=on_call_tool)
