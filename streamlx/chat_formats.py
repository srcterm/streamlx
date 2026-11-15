"""Chat formats: what a server needs to know about a model's prompts and
replies that mlx-lm does not get from the checkpoint.

Prompt formats. mlx-lm renders a chat with the checkpoint's Jinja template
and parses tool calls with a parser it picks from that template. A checkpoint
without one (DeepSeek publishes a Python encoder for V4.1, and its exports
carry neither) is prompted in a fallback format and returns its tool calls as
text. A format here is a prompt renderer and a tool-call parser for one model
type, handed to mlx-lm's tokenizer loader as that model's defaults: the hook
mlx-lm's own Python templates use, so a template in the checkpoint still
wins.

    load_tokenizer(path, defaults=chat_formats.defaults(model_type))

Formats: deepseek_v41.

Reply framing (reply_framing). A model that writes its own message framing
(Inkling 2) describes it in its tokenizer's response_template; mlx-lm's
response format is made from that, so a server sorts a reply into reasoning,
content and tool calls.

generation_suffix: the tokens a template appends to ask for an answer, where
a prompt's history ends.
"""

from __future__ import annotations

import json
import re
import sys
import types

# -- DeepSeek-V4.1 ------------------------------------------------------------
# The format of DeepSeek's encoder (encoding/encoding.py in
# deepseek-ai/DeepSeek-V4.1-Flash-1017, MIT): system text opens with
# <｜System｜>, a thinking prompt with its effort line; there is no tool role,
# a tool's result is a <tool_result> block of the user turn, in the order of
# the calls it answers; tool calls are a "<｜DSML｜ calls>" block. Covers system,
# user, assistant and tool messages, tools and thinking; not images, tasks,
# tool namespaces or response_format. The tools paragraph is DeepSeek's text.
_BOS, _EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
_SYSTEM, _USER = "<｜System｜>", "<｜User｜>"
_ASSISTANT = "<｜Assistant｜>"
_THINK, _THINK_END = "<think>", "</think>"
_CALLS, _INVOKE = "｜DSML｜ calls", "｜DSML｜ invoke"
_PARAM = "｜DSML｜ parameter"
_EFFORT = {"low": 50, "high": 75, "max": 100}
_V41_TOOLS = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{calls}>" block like the following:

<{calls}>
<{invoke} name="$TOOL_NAME">
<{param} name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{param}>
...
</{invoke}>
<{invoke} name="$TOOL_NAME2">
...
</{invoke}>
</{calls}>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {think}), you MUST output your complete reasoning inside {think}...{think_end} BEFORE any tool calls or final response.

Otherwise, output directly after {think_end} with tool calls or final response.

### Available Tool Schemas

{schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


def _json(v) -> str:
    return json.dumps(v, ensure_ascii=False)


def _text(content) -> str:
    if isinstance(content, list):   # OpenAI content parts
        return "\n\n".join(
            p.get("text", "") if p.get("type") == "text"
            else f"[Unsupported {p.get('type')}]" for p in content)
    return content or ""


def _v41_turns(messages) -> list:
    """Consecutive user and tool messages as one user turn of text and
    tool-result blocks; a turn's results in the order of their calls."""
    out, order = [], {}
    for m in messages:
        role = m.get("role")
        if role in ("user", "tool"):
            text = _text(m.get("content"))
            block = (("tool", m.get("tool_call_id", ""), text)
                     if role == "tool" else ("text", "", text))
            if out and out[-1]["role"] == "user":
                out[-1]["blocks"].append(block)
            else:
                out.append({"role": "user", "blocks": [block], "order": order})
            continue
        out.append(dict(m))
        if role == "assistant" and m.get("tool_calls"):
            order = {tc.get("id"): i for i, tc in enumerate(m["tool_calls"])
                     if tc.get("id")}
    for t in out:
        if t["role"] != "user":
            continue
        results = sorted((b for b in t["blocks"] if b[0] == "tool"),
                         key=lambda b: t["order"].get(b[1], 0))
        if len(results) > 1 and t["order"]:
            it = iter(results)
            t["blocks"] = [next(it) if b[0] == "tool" else b
                           for b in t["blocks"]]
    return out


def _v41_calls(tool_calls) -> str:
    out = []
    for tc in tool_calls:
        f = tc.get("function", tc)
        args = f.get("arguments")
        for _ in range(2):          # a JSON string, even one encoded twice
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    break
        if not isinstance(args, dict):
            args = {"arguments": args}
        params = "\n".join(
            f'<{_PARAM} name="{k}" string="{str(isinstance(v, str)).lower()}">'
            f'{v if isinstance(v, str) else _json(v)}</{_PARAM}>'
            for k, v in args.items())
        out.append(f'<{_INVOKE} name="{f["name"]}">\n{params}\n</{_INVOKE}>')
    return f"\n\n<{_CALLS}>\n" + "\n".join(out) + f"\n</{_CALLS}>"


def render_v41(messages, tools=None, thinking: bool = False, effort=None,
               add_generation_prompt: bool = True) -> str:
    """OpenAI chat messages as DeepSeek-V4.1's prompt text. effort: 1-100 or
    low / high / max (thinking prompts only; default high)."""
    turns = _v41_turns(messages)
    if tools:                       # rendered with the opening system text
        if not turns or turns[0]["role"] != "system":
            turns.insert(0, {"role": "system", "content": ""})
        turns[0]["tools"] = tools
    effort = _EFFORT["high"] if effort is None else _EFFORT.get(effort, effort)
    if not (isinstance(effort, int) and 1 <= effort <= 100):
        raise ValueError(f"reasoning effort {effort!r}: 1-100 or "
                         f"{' / '.join(_EFFORT)}")
    # with tools every past turn keeps its thinking
    drop = not any(t.get("tools") for t in turns)
    last_user = max(
        (i for i, t in enumerate(turns)
         if (t["role"] == "user" and any(b[0] == "text" for b in t["blocks"]))
         or (t["role"] == "system" and i > 0)), default=-1)
    out = _BOS
    for i, t in enumerate(turns):
        role = t["role"]
        if i == 0 and (thinking or role == "system"):
            out += _SYSTEM
        if i == 0 and thinking:
            out += (f"Reasoning Effort: {effort} (range 1-100, the higher the "
                    "value, the more thorough the reasoning)\n\n")
        if role == "system":
            out += (_SYSTEM if i else "") + _text(t.get("content"))
            if t.get("tools"):
                out += "\n\n" + _V41_TOOLS.format(
                    calls=_CALLS, invoke=_INVOKE, param=_PARAM, think=_THINK,
                    think_end=_THINK_END, schemas="\n".join(
                        _json(x.get("function", x)) for x in t["tools"]))
        elif role == "user":
            out += _USER + "\n\n".join(
                b[2] if b[0] == "text"
                else f"<tool_result>{b[2]}</tool_result>"
                for b in t["blocks"])
        elif role == "assistant":
            if thinking and (not drop or i > last_user):
                out += (t.get("reasoning_content") or t.get("reasoning")
                        or "") + _THINK_END
            out += _text(t.get("content"))
            if t.get("tool_calls"):
                out += _v41_calls(t["tool_calls"])
            out += _EOS
        elif role == "latest_reminder":
            out += "<｜latest_reminder｜>" + _text(t.get("content"))
        else:
            raise NotImplementedError(f"deepseek_v41: message role {role!r}")
        last = i + 1 == len(turns)
        answered = last or turns[i + 1]["role"] in ("assistant",
                                                    "latest_reminder")
        if (answered and (role == "user" or (role == "system" and i > 0))
                and (add_generation_prompt or not last)):
            out += _ASSISTANT + (
                _THINK if thinking and (not drop or i >= last_user)
                else _THINK_END)
    return out


def _v41_template(messages, tools=None, add_generation_prompt=False,
                  continue_final_message=False, enable_thinking=False,
                  reasoning_effort=None, **_) -> str:
    out = render_v41(list(messages), tools, bool(enable_thinking),
                     reasoning_effort, add_generation_prompt)
    if (continue_final_message and messages
            and messages[-1].get("role") == "assistant"):
        out = out.removesuffix(_EOS)
    return out


_V41_INVOKE = re.compile(
    r'<｜DSML｜ invoke name="([^"]+)">(.*?)</｜DSML｜ invoke>', re.DOTALL)
_V41_PARAM = re.compile(
    r'<｜DSML｜ parameter name="([^"]+)"(?: string="(true|false)")?>'
    r'(.*?)</｜DSML｜ parameter>', re.DOTALL)


def parse_v41_calls(text: str, tools=None):
    """The inside of a "<｜DSML｜ calls>" block as {"name", "arguments"}, a
    list of them for several calls."""
    def value(raw, is_string):
        if is_string == "true":
            return raw
        try:
            return json.loads(raw)
        except ValueError:
            return raw

    calls = [{"name": name, "arguments": {k: value(v, s) for k, s, v
                                          in _V41_PARAM.findall(body)}}
             for name, body in _V41_INVOKE.findall(text)]
    if not calls:
        raise ValueError("no <｜DSML｜ invoke> block")
    return calls[0] if len(calls) == 1 else calls


# -- registry -----------------------------------------------------------------
# model type -> (chat template, tool-call start, tool-call end, parser)
_FORMATS = {
    "deepseek_v41": (_v41_template, f"<{_CALLS}>", f"</{_CALLS}>",
                     parse_v41_calls),
}


def defaults(model_type: str) -> dict:
    """mlx-lm tokenizer-loader defaults that select this module's format for
    `model_type`; {} when it has none. mlx-lm looks both up as modules of its
    own packages, so they are registered there under a streamlx name."""
    fmt = _FORMATS.get(model_type)
    if fmt is None:
        return {}
    import mlx_lm.chat_templates  # the parent packages
    import mlx_lm.tool_parsers  # noqa: F401

    name = f"streamlx_{model_type}"
    template, start, end, parser = fmt
    tmpl = types.ModuleType(f"mlx_lm.chat_templates.{name}")
    tmpl.apply_chat_template = template
    tool = types.ModuleType(f"mlx_lm.tool_parsers.{name}")
    tool.tool_call_start, tool.tool_call_end = start, end
    tool.parse_tool_call = parser
    sys.modules.setdefault(tmpl.__name__, tmpl)
    sys.modules.setdefault(tool.__name__, tool)
    return {"chat_template_type": name, "tool_parser_type": name}


# -- what a template says about replies ----------------------------------------
def generation_suffix(tokenizer, template_args: dict) -> list[int]:
    """The tokens the chat template appends for add_generation_prompt, e.g.
    Qwen3.6's "<|im_start|>assistant\\n<think>\\n" ([] if none)."""
    msgs = [{"role": "user", "content": "x"}]
    try:
        a, b = (list(tokenizer.apply_chat_template(
            msgs, add_generation_prompt=g, tokenize=True, **template_args))
            for g in (False, True))
    except Exception:  # noqa: BLE001 — no template, or one needing more
        return []
    return b[len(a):] if b[:len(a)] == a else []


def reply_framing(tokenizer) -> bool:
    """A model that writes its own message framing (inkling2: its template
    has no generation prompt; the reply opens each part with a content-type
    token and closes it with an end token) describes it in tokenizer_config's
    `response_template`: fields with an open pattern and a close. Give the
    tokenizer mlx-lm's response format made from it, unless it has one: the
    server's text state machine then sorts the reply into reasoning, content
    and tool calls, and drops the framing. Returns whether it did."""
    tpl = tokenizer.init_kwargs.get("response_template")
    if not isinstance(tpl, dict) or tokenizer.response_transitions:
        return False
    states = {"reasoning_content": "reasoning", "content": "normal",
              "tool_calls": "tool"}
    added = list(tokenizer.get_added_vocab())
    trans, tool = {"header": []}, None
    for name, field in (tpl.get("fields") or {}).items():
        state, close = states.get(name), field.get("close")
        # a part opens with a token the pattern matches by itself (what may
        # precede it, a message header, is dropped as header text)
        opens = [field["open"]] if field.get("open") else []
        if field.get("open_pattern"):
            pattern = re.compile(field["open_pattern"])
            opens = [t for t in added if pattern.fullmatch(t)]
        if state is None or not close or not opens:
            continue
        trans["header"] += [(t, state) for t in opens]
        trans.setdefault(state, []).append((close, "header"))
        if state == "tool":
            tool = (field, opens[0])
    if len(trans) < 2:
        return False
    tokenizer._response_format = {"initial_state": "header",
                                  "transitions": trans}
    if tool is not None:
        field, start = tool
        shape = (field.get("transform") or {}).get("function") or {
            "name": "{content.name}", "arguments": "{content.arguments}"}

        def fill(x, content):
            """`{content.a.b}` -> that value of the parsed call."""
            if isinstance(x, dict):
                return {k: fill(v, content) for k, v in x.items()}
            m = re.fullmatch(r"\{content((?:\.\w+)*)\}", x) \
                if isinstance(x, str) else None
            if m is None:
                return x
            for key in filter(None, m.group(1).split(".")):
                content = content[key]
            return content

        def parse_tool_call(text, tools=None):
            content = json.loads(text) if field.get("content") == "json" \
                else text
            call = fill(shape, content)
            return {"name": call["name"], "arguments": call["arguments"]}

        tokenizer._tool_parser = parse_tool_call
        tokenizer._tool_call_start = start   # has_tool_calling
        tokenizer._tool_call_end = None      # the response format closes it
    return True
