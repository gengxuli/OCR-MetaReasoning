import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, List, Optional, Tuple


FINAL_PATTERNS = [
    re.compile(
        r"^\s*(?:#{1,6}\s*)?(?:[-*]\s*)?(?:\*\*)?(?:final\s+answer|answer|答案|最终答案)\s*(?:\*\*)?\s*[:：]\s*(.+)$",
        re.IGNORECASE,
    )
]

_CURRENCY_SYMBOL_TO_CODE = {
    "$": "usd",
    "€": "eur",
    "£": "gbp",
    "¥": "jpy",
    "₹": "inr",
}

_CURRENCY_CODE_RE = r"(?:usd|eur|gbp|jpy|inr|vnd|rm|rs\.?)"
_CURRENCY_SYMBOL_RE = r"[$€£¥₹]"
_NUMBER_BODY_RE = r"[-+]?(?:(?:\d{1,3}(?:[,\s]\d{2,3})+)|\d+)(?:\.\d+)?|[-+]?\.\d+"
_STRICT_NUMBER_RE = re.compile(
    rf"^\s*(?:{_CURRENCY_SYMBOL_RE}|{_CURRENCY_CODE_RE})?\s*"
    rf"(?P<number>{_NUMBER_BODY_RE})"
    rf"\s*(?:{_CURRENCY_SYMBOL_RE}|{_CURRENCY_CODE_RE})?\s*%?\s*$",
    re.IGNORECASE,
)
_NUMERIC_UNIT_RE = (
    r"(?:"
    r"(?:us\s*)?dollars?|rupees?|rs\.?|vnd|rm|(?:italian\s+)?lire|euros?|pounds?|"
    r"lakhs?|lacs?|crores?|millions?|billions?|thousands?|"
    r"shares?|meetings?|days?|points?|percent(?:age)?(?:\s+points?)?|"
    r"ppm|parts\s+per\s+million|kcal/l|mg/kg/d\.?|g/24\s*hr|gtc|local\s+currency"
    r")"
)
_NUMBER_WITH_UNIT_RE = re.compile(
    rf"^\s*(?P<number_part>(?:{_CURRENCY_SYMBOL_RE}|{_CURRENCY_CODE_RE})?\s*"
    rf"(?:{_NUMBER_BODY_RE})"
    rf"\s*(?:{_CURRENCY_SYMBOL_RE}|{_CURRENCY_CODE_RE})?\s*%?)"
    rf"\s+{_NUMERIC_UNIT_RE}\s*$",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"^\d{2,4}\s*[-/]\s*\d{2,4}(?:\s*[-/]\s*\d{2,4})?$")
_LATEX_BRACED_COMMANDS = {
    "boxed",
    "boldsymbol",
    "mathbf",
    "mathrm",
    "text",
    "textbf",
}
_STRING_SUFFIX_WORDS = (
    "answer",
    "category",
    "class",
    "code",
    "field",
    "generation",
    "group",
    "item",
    "label",
    "level",
    "line",
    "month",
    "option",
    "phrase",
    "region",
    "row",
    "section",
    "season",
    "sign",
    "state",
    "step",
    "sub-class",
    "subclass",
    "symbol",
    "word",
    "year",
)
_STRING_PREFIX_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^(?:the\s+)?(?:exact\s+phrase|correct\s+phone\s+number|hidden\s+id\s+code\s+text|"
        r"missing\s+symbol|removed\s+item|refunded\s+line\s+item|double-counted\s+item|"
        r"responsible\s+state|ability|group|demographic\s+group)\s+"
        r"(?:is|are|was|were|must\s+be|:)\s+(.+)$",
        r"^the\s+customer\s+returned\s+(.+)$",
        r"^the\s+analyst\s+is\s+referring\s+to\s+(.+)$",
        r"^the\s+researcher\s+is\s+analyzing\s+(.+)$",
        r"^the\s+rows\s+are\s+sorted.*?\s+by\s+(.+)$",
    )
]


def normalize_string(value: Any) -> str:
    text = strip_outer_formatting("" if value is None else str(value))
    text = text.strip().lower()
    text = text.replace("％", "%")
    text = text.replace(r"\%", "%")
    text = text.replace(r"\$", "$")
    text = text.replace("，", ",")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n\"'")

    if _DATE_RE.fullmatch(text):
        return "date:" + re.sub(r"\s*[-/]\s*", "/", text)

    if re.fullmatch(rf"(?:{_NUMBER_BODY_RE})\s*%", text):
        number = normalize_number(text)
        if number is not None:
            return f"percent:{number}"

    currency_code = detect_currency_code(text)
    if currency_code:
        number = normalize_number(text)
        if number is not None:
            return f"currency:{currency_code}:{number}"

    return text


def normalize_number(value: Any) -> Optional[str]:
    text = strip_outer_formatting("" if value is None else str(value))
    text = text.strip()
    text = text.replace("％", "%")
    text = text.replace(r"\%", "%")
    text = text.replace(r"\$", "$")
    text = text.replace("−", "-")
    text = text.replace("–", "-").replace("—", "-")

    if text.startswith("(") and text.endswith(")"):
        inner = normalize_number(text[1:-1])
        if inner is not None:
            return inner if inner.startswith("-") else f"-{inner}"

    text = strip_trailing_parenthetical(text)

    if text.endswith("."):
        without_period = text[:-1].rstrip()
        if _STRICT_NUMBER_RE.fullmatch(without_period):
            text = without_period

    match = _STRICT_NUMBER_RE.fullmatch(text)
    if not match:
        return None

    number_text = re.sub(r"[,\s]", "", match.group("number"))
    try:
        number = Decimal(number_text)
    except InvalidOperation:
        return None

    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized == "-0" else normalized


def strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|JSON|text|TEXT)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def strip_outer_formatting(text: str) -> str:
    text = strip_code_fence(text)
    wrappers = [
        ("**", "**"),
        ("__", "__"),
        ("*", "*"),
        ("`", "`"),
        ('"', '"'),
        ("'", "'"),
        ("“", "”"),
        ("‘", "’"),
    ]

    changed = True
    while changed:
        changed = False
        text = text.strip()
        for marker in ("**", "__", "`"):
            if text.endswith(marker) and not text.startswith(marker):
                text = text[: -len(marker)].strip()
                changed = True
                break
            if text.startswith(marker) and not text.endswith(marker):
                text = text[len(marker) :].strip()
                changed = True
                break
        if changed:
            continue

        latex_inner = strip_latex_wrapper(text)
        if latex_inner is not None:
            text = latex_inner
            changed = True
            continue

        for left, right in wrappers:
            if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
                text = text[len(left) : len(text) - len(right)].strip()
                changed = True
                break
    return text.strip()


def strip_latex_wrapper(text: str) -> Optional[str]:
    for left, right in (("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
            return text[len(left) : len(text) - len(right)].strip()

    return unwrap_latex_command(text)


def unwrap_latex_command(text: str) -> Optional[str]:
    match = re.match(r"^\\(?P<command>[A-Za-z]+)\s*\{", text)
    if not match or match.group("command") not in _LATEX_BRACED_COMMANDS:
        return None

    open_index = match.end() - 1
    close_index = find_matching_brace(text, open_index)
    if close_index == len(text) - 1:
        return text[open_index + 1 : close_index].strip()
    return None


def find_matching_brace(text: str, open_index: int) -> int:
    depth = 0
    escaped = False
    for index in range(open_index, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def strip_trailing_parenthetical(text: str) -> str:
    text = text.strip()
    while True:
        match = re.search(r"\s+\([^()]*\)\s*$", text)
        if not match:
            return text
        text = text[: match.start()].strip()


def detect_currency_code(text: str) -> Optional[str]:
    for symbol, code in _CURRENCY_SYMBOL_TO_CODE.items():
        if symbol in text:
            return code

    lowered = text.lower()
    for code in ("usd", "eur", "gbp", "jpy", "inr", "vnd", "rm"):
        if re.search(rf"\b{code}\b", lowered):
            return code
    if re.search(r"\brs\.?\b", lowered):
        return "inr"
    return None


def extract_final_answer(output: str, answer_type: str) -> str:
    if not output:
        return ""

    clean_output = output.strip()
    lines = [line.strip() for line in clean_output.splitlines() if line.strip()]

    for line in reversed(lines):
        for pattern in FINAL_PATTERNS:
            match = pattern.search(line)
            if match:
                return clean_extracted_answer(match.group(1), answer_type)

    if answer_type == "json":
        json_text = extract_whole_json_text(clean_output)
        if json_text:
            return clean_extracted_answer(json_text, answer_type)

    return ""


def clean_extracted_answer(answer: str, answer_type: str) -> str:
    answer = strip_outer_formatting(answer)
    answer = strip_trailing_answer_punctuation(answer, answer_type)
    if answer_type in {"integer", "float"} and answer.endswith("."):
        without_period = answer[:-1].rstrip()
        if normalize_number(without_period) is not None:
            return without_period
    return answer


def strip_trailing_answer_punctuation(answer: str, answer_type: str) -> str:
    answer = answer.strip()
    if answer_type == "json":
        return answer
    changed = True
    while changed:
        changed = False
        stripped = answer.rstrip()
        for marker in ("**", "__", "`"):
            if stripped.endswith(marker) and not stripped.startswith(marker):
                stripped = stripped[: -len(marker)].rstrip()
                changed = True
        if stripped != answer:
            answer = stripped
            continue
        if answer.endswith(("。", ";")):
            answer = answer[:-1].rstrip()
            changed = True
    return answer


def iter_latex_command_contents(text: str) -> Iterable[str]:
    for match in re.finditer(r"\\(?P<command>[A-Za-z]+)\s*\{", text):
        if match.group("command") not in _LATEX_BRACED_COMMANDS:
            continue
        open_index = match.end() - 1
        close_index = find_matching_brace(text, open_index)
        if close_index > open_index:
            yield text[open_index + 1 : close_index].strip()


def build_numeric_candidates(value: Any) -> List[str]:
    text = "" if value is None else str(value)
    candidates: List[str] = [text, strip_trailing_parenthetical(text)]

    candidates.extend(iter_latex_command_contents(text))
    candidates.extend(match.group(1) for match in re.finditer(r"\*\*(.+?)\*\*", text))
    candidates.extend(match.group(1) for match in re.finditer(r"`([^`]+)`", text))

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        cleaned = strip_outer_formatting(candidate)
        if cleaned not in seen:
            seen.add(cleaned)
            unique_candidates.append(cleaned)
    return unique_candidates


def score_numeric_prediction(prediction: str, target: str) -> float:
    target_number = normalize_number(target)
    if target_number is None:
        return 0.0

    pred_number = normalize_number(prediction)
    return 1.0 if pred_number is not None and pred_number == target_number else 0.0


def build_string_candidates(value: Any) -> List[str]:
    text = "" if value is None else str(value)
    queue = [
        text,
        *iter_latex_command_contents(text),
        *(match.group(1) for match in re.finditer(r"\*\*(.+?)\*\*", text)),
        *(match.group(1) for match in re.finditer(r"`([^`]+)`", text)),
        *(match.group(1) for match in re.finditer(r"\(([^()]+)\)", text)),
    ]
    seen = set()
    candidates: List[str] = []

    def add(candidate: str) -> None:
        candidate = strip_outer_formatting(candidate).strip()
        if not candidate or candidate in seen:
            return
        seen.add(candidate)
        candidates.append(candidate)
        queue.append(candidate)

    while queue and len(seen) < 100:
        candidate = queue.pop(0)
        cleaned = strip_outer_formatting(candidate).strip()
        if not cleaned or cleaned in seen:
            continue

        seen.add(cleaned)
        candidates.append(cleaned)

        if cleaned.endswith((".", "。")):
            add(cleaned[:-1].rstrip())

        without_parenthetical = strip_trailing_parenthetical(cleaned)
        if without_parenthetical != cleaned:
            add(without_parenthetical)

        for pattern in _STRING_PREFIX_PATTERNS:
            match = pattern.match(cleaned)
            if match:
                add(match.group(1))

        subject_match = re.match(
            r"^(.+?)\s+(?:is|are|was|were)\s+(?P<predicate>(?:the|a|an|one)\b.+)$",
            cleaned,
            re.IGNORECASE,
        )
        if subject_match and not re.search(
            r"\b(?:wrong|incorrect|not|fails?|does\s+not|isn't|aren't)\b",
            subject_match.group("predicate"),
            re.IGNORECASE,
        ):
            add(subject_match.group(1))

        suffix_match = re.match(
            rf"^(?:the\s+)?(.+?)\s+(?:{'|'.join(re.escape(word) for word in _STRING_SUFFIX_WORDS)})$",
            cleaned,
            re.IGNORECASE,
        )
        if suffix_match:
            add(suffix_match.group(1))

        explanatory_match = re.match(
            r"^(.+?),\s+(?:caused\s+by|which|matching|as\s+labeled|as\s+shown|where|representing|the\s+only|this\s+is)\b.+$",
            cleaned,
            re.IGNORECASE,
        )
        if explanatory_match:
            add(explanatory_match.group(1))

        under_match = re.match(r"^(.+?)\s+under\s+.+$", cleaned, re.IGNORECASE)
        if under_match:
            add(under_match.group(1))

    return candidates


def score_string_prediction(prediction: str, target: str) -> float:
    return 1.0 if normalize_string(prediction) == normalize_string(target) else 0.0


def extract_whole_json_text(text: str) -> str:
    stripped = strip_outer_formatting(text)
    try:
        json.loads(stripped)
        return stripped
    except Exception:
        return ""


def extract_json_text(text: str) -> str:
    fenced = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    stripped = strip_outer_formatting(text)
    try:
        json.loads(stripped)
        return stripped
    except Exception:
        pass

    for start, char in enumerate(text):
        if char not in "[{":
            continue
        candidate = find_balanced_json(text, start)
        if not candidate:
            continue
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            continue
    return ""


def find_balanced_json(text: str, start: int) -> str:
    opener = text[start]
    expected_closer = "}" if opener == "{" else "]"
    stack = [expected_closer]
    in_string = False
    escaped = False

    for index in range(start + 1, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char in "{[":
            stack.append("}" if char == "{" else "]")
            continue
        if char in "}]":
            if not stack or char != stack[-1]:
                return ""
            stack.pop()
            if not stack:
                return text[start : index + 1].strip()

    return ""


def flatten_json(data: Any, prefix: str = "") -> List[Tuple[str, str]]:
    if isinstance(data, dict):
        items: List[Tuple[str, str]] = []
        for key in sorted(data):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.extend(flatten_json(data[key], next_prefix))
        return items
    if isinstance(data, list):
        items = []
        for index, value in enumerate(data):
            items.extend(flatten_json(value, f"{prefix}[{index}]"))
        return items
    return [(prefix, normalize_string(data))]


def score_json_f1(prediction: str, target: str) -> float:
    try:
        pred_obj = json.loads(strip_outer_formatting(prediction))
        target_obj = json.loads(strip_outer_formatting(target))
    except Exception:
        return 0.0

    pred_items = set(flatten_json(pred_obj))
    target_items = set(flatten_json(target_obj))
    if not pred_items and not target_items:
        return 1.0
    if not pred_items or not target_items:
        return 0.0

    true_positive = len(pred_items & target_items)
    precision = true_positive / len(pred_items)
    recall = true_positive / len(target_items)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_prediction(prediction: str, target: str, metric: str, answer_type: str) -> float:
    if metric == "numeric" or answer_type in {"integer", "float"}:
        return score_numeric_prediction(prediction, target)
    if metric == "json_f1" or answer_type == "json":
        return score_json_f1(prediction, target)
    return score_string_prediction(prediction, target)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
