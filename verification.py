"""Schema + semantic intent + safety invariants; never execute during verification."""
from jsonschema import Draft202012Validator
from taxonomy import Fault
import json
import traceback


def _trace_for_validation(error):
    """Generate an actual Python traceback for the raw-stack-trace feedback arm."""
    try:
        raise error
    except Exception:
        return traceback.format_exc()


def parse_proposal(raw):
    if not isinstance(raw, str):
        return None, Fault("malformed_payload", "not_text", "LLM proposal was not a string", actual=type(raw).__name__)
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        return None, Fault("malformed_payload", "invalid_json", str(exc), expected='JSON {"tool":string,"args":object}',actual=raw[:180], trace="".join(traceback.format_exception(exc)))
    if not isinstance(value, dict) or not isinstance(value.get("tool"), str) or not isinstance(value.get("args"), dict):
        return None, Fault("malformed_payload", "bad_shape", "tool must be a string and args an object", expected='{"tool":string,"args":object}',actual=repr(value)[:180])
    return {"tool": value["tool"], "args": value["args"]}, None




def _grounding_context(step, history):
    grounding = step.get("grounding") or {}
    argument = grounding.get("argument")
    history_field = grounding.get("history_field")
    if not argument or not history_field:
        return grounding, argument, None
    values = []
    for item in history or ():
        output = item.get("output") if isinstance(item, dict) else None
        if isinstance(output, dict) and history_field in output:
            values.append(output[history_field])
    return grounding, argument, (values[-1] if values else None)


def _suggest_tool_for_argument(catalog, argument):
    if not argument:
        return ""
    candidates = []
    for name, tool in catalog.items():
        schema = getattr(tool, "input_schema", {}) or {}
        properties = schema.get("properties", {}) or {}
        if argument in properties:
            candidates.append(name)
    return candidates[0] if len(candidates) == 1 else ""

def validate(call, catalog, step, *, history=(), schema_enabled=True, safety_enabled=True):
    grounding, grounded_argument, grounded_value = _grounding_context(step, history)
    evidence_suggested_tool = _suggest_tool_for_argument(catalog, grounded_argument)

    if call["tool"] not in catalog:
        return Fault(
            "incorrect_tool", "unknown_tool", f"Tool {call['tool']!r} does not exist",
            expected=f"one of {sorted(catalog)}", actual=call["tool"],
            suggested_tool=evidence_suggested_tool or step.get("expected_tool", ""),
        )
    if schema_enabled:
        schema = catalog[call["tool"]].input_schema
        errors = sorted(Draft202012Validator(schema).iter_errors(call["args"]), key=lambda e: (list(map(str,e.path)),e.message))
        if errors:
            err = errors[0]
            field = ".".join(str(x) for x in err.path)
            if err.validator == "required":
                try:
                    field = str(err.message).split("'")[1]
                except IndexError:
                    field = "required field"
            expected = repr(err.validator_value)
            message = err.message
            selected_properties = (schema.get("properties", {}) or {})
            if grounded_argument and grounded_value is not None:
                # Ground corrective feedback in an ACTUAL prior MCP result, not hidden task truth.
                # This gives the model the information needed to repair a corrupted argument.
                if field == grounded_argument or grounded_argument not in selected_properties:
                    message += (
                        f". Previous MCP evidence supplies {grounded_argument}="
                        f"{grounded_value!r}; use that retrieved value for the next dependent call"
                    )
                    if field == grounded_argument:
                        expected += f"; grounded value {grounded_value!r}"
            category = "parameter_type_mismatch" if err.validator == "type" else "schema_mismatch"
            suggested = evidence_suggested_tool if evidence_suggested_tool != call["tool"] else ""
            return Fault(
                category, f"jsonschema_{err.validator}", message, field=field,
                expected=expected, actual=repr(call["args"]), suggested_tool=suggested,
                trace=_trace_for_validation(err),
            )
    if safety_enabled:
        # An intent check is independent of schema validity. Prevent valid, wrong-tool actions.
        wanted = step.get("expected_tool", "")
        if wanted and call["tool"] != wanted:
            return Fault("incorrect_tool", "semantic_tool_selection",f"Selected {call['tool']}, but the current task requires {wanted}", expected=wanted,actual=call["tool"],suggested_tool=wanted)
        if call["tool"] == "reserve_stock":
            qty = call["args"].get("quantity", None)
            if type(qty) is not int or qty <= 0 or qty > 100:
                return Fault("safety_violation", "unsafe_quantity", "Reservations must use a positive integer quantity no more than 100", field="quantity",expected="integer 1..100",actual=repr(qty))
        # Evidence-grounding checks are allowed because they validate provenance
        # against a previous MCP result rather than against a hidden answer key.
        if grounding:
            argument = grounded_argument
            history_field = grounding.get("history_field")
            mode = grounding.get("mode", "equals")
            if grounded_value is None:
                return Fault(
                    "semantic_failure", "missing_reference_evidence",
                    f"No previous MCP result supplies required evidence field {history_field!r}",
                    field=argument, expected=f"value retrieved from prior MCP field {history_field}",
                    actual=repr(call["args"].get(argument)),
                    suggested_tool=evidence_suggested_tool,
                )
            expected_value = grounded_value
            actual_value = call["args"].get(argument)
            grounded = actual_value == expected_value
            if mode == "same_set" and isinstance(actual_value, list) and isinstance(expected_value, list):
                grounded = set(actual_value) == set(expected_value)
            if not grounded:
                return Fault(
                    "semantic_failure", "ungrounded_argument",
                    f"Argument {argument!r} is not grounded in the previous MCP evidence",
                    field=argument, expected=repr(expected_value), actual=repr(actual_value),
                    suggested_tool=(evidence_suggested_tool if evidence_suggested_tool != call["tool"] else ""),
                )
        # Do not compare with task ground-truth arguments: that would leak the
        # answer into verification and unfairly inflate the closed-loop score.
    return None


def unwrap_mcp_reply(result):
    if getattr(result, "is_error", False):
        message = " | ".join(getattr(item,"text",str(item)) for item in (result.content or []))
        return None, Fault("execution_failure", "mcp_tool_error",message or "MCP tool returned an error",trace=message)
    data = getattr(result, "structured_content", None)
    if isinstance(data, dict) and set(data) == {"result"}:
        data = data["result"]
    if data is None:
        text = " ".join(str(getattr(item,"text",item)) for item in (getattr(result,"content",[]) or []))
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            data = {"text": text}
    return data, None
