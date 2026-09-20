"""LangGraph agent over the H-1B sponsorship data.

Graph:  agent -> (tool_calls?) -> tools -> agent -> ... -> END

The agent is deliberately tool-bound rather than free-form: every number in an
answer has to come back from one of these tools, which is what keeps it from
inventing sponsorship statistics.
"""
import os, json
from typing import Annotated, TypedDict, Optional, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from . import db

# Provider-agnostic. Whichever key is present wins, so the same deployment runs
# on OpenAI, Gemini or Anthropic without a code change.
_PROVIDERS = [
    ("openai",    ("OPENAI_API_KEY",),                  "gpt-5-mini"),
    ("gemini",    ("GOOGLE_API_KEY", "GEMINI_API_KEY"), "gemini-2.5-flash"),
    ("anthropic", ("ANTHROPIC_API_KEY",),               "claude-sonnet-4-5-20250929"),
]


def _detect():
    forced = os.environ.get("RADAR_PROVIDER")
    for name, envs, default in _PROVIDERS:
        key = next((os.environ[e] for e in envs if os.environ.get(e)), None)
        if forced == name or (not forced and key):
            return name, key, os.environ.get("RADAR_MODEL") or default
    return None, None, None


PROVIDER, _KEY, MODEL = _detect()


def available():
    return PROVIDER is not None and _KEY is not None


SYSTEM = """You are Sponsorship Radar, an assistant for international students (F-1/OPT)
who need to know which US employers actually sponsor work visas.

You answer from one dataset: US Department of Labor disclosure files.
  - LCA / H-1B filings, FY2023 through FY2026 Q1 (quarter ending Dec 2025)
  - PERM green-card filings, FY2024 through FY2026 Q1

Rules you must follow:
1. NEVER state a sponsorship number you did not get from a tool call. If a tool
   returns nothing, say the employer is not in the data rather than guessing.
2. An LCA is a *labor condition application* - the employer's step 1 of sponsoring.
   It is not a granted visa and not a lottery win. Employers file more LCAs than
   people they hire. Say "filings" or "positions", never "visas granted".
3. The number that actually matters to a job seeker is outside hires
   (NEW_EMPLOYMENT + CHANGE_EMPLOYER), not total filings. A company with thousands
   of renewals and few outside hires is not realistically hiring from outside.
   Lead with that distinction when it changes the answer.
4. Flag these when true: h1b_dependent = Yes (IT staffing/consulting - they sponsor
   readily but conditions vary), willful_violator = Yes (rule violations),
   cap_exempt_likely = true (university/hospital - can sponsor outside the March
   lottery, which is the single most useful fact for a student who missed the cap).
5. FY2026 contains only Q1 (Oct-Dec 2025). Never compare a partial FY2026 against
   a full year without saying so.
6. Be concrete and brief. Give numbers, then what to do about them.

Tool choice:
  - a named company            -> lookup_company
  - several named companies    -> compare_companies
  - role / place / pay filters -> shortlist_sponsors (cap_exempt=True if they
                                  missed the lottery or ask about universities)
  - a *kind* of company        -> semantic_search
  - rankings, trends, anything -> query_data (read-only SQL)
    the fixed tools can't express

You cannot predict whether any individual will get a visa. You report employer behaviour."""


# ----------------------------------------------------------------- tools
@tool
def lookup_company(name: str) -> str:
    """Get the full H-1B/green-card sponsorship profile for one company by name.
    Use this whenever the user names a specific employer."""
    p = db.employer_profile(name)
    if not p.get("found"):
        return json.dumps({"found": False, "message": "No employer matching %r in DOL data." % name})
    return json.dumps(p, default=str)


@tool
def shortlist_sponsors(role: Optional[str] = None, state: Optional[str] = None,
                       city: Optional[str] = None, min_wage: Optional[int] = None,
                       cap_exempt: Optional[bool] = None, exclude_staffing: bool = False,
                       sort: str = "score", limit: int = 20) -> str:
    """Find employers that sponsor a given ROLE in a given STATE/CITY above MIN_WAGE.
    role matches job titles and SOC occupation titles (e.g. 'Software Developer',
    'Data Scientist'). state is a 2-letter code. Set cap_exempt=True for
    universities/hospitals that can sponsor outside the H-1B lottery.
    Set exclude_staffing=True to drop H-1B-dependent consulting firms.
    sort: 'score' | 'volume' | 'wage' | 'recent'."""
    return json.dumps(db.search_employers(role=role, state=state, city=city,
        min_wage=min_wage, cap_exempt=cap_exempt, exclude_dependent=exclude_staffing,
        sort=sort, limit=min(limit, 40)), default=str)


@tool
def compare_companies(names: List[str]) -> str:
    """Compare sponsorship behaviour across several named companies side by side."""
    out = []
    for n in names[:6]:
        p = db.employer_profile(n)
        if p.get("found"):
            out.append({k: p.get(k) for k in ("display_name", "lca_total", "recent_filings",
                "recent_outside", "outside_ratio", "wage_median", "perm_certified",
                "cap_exempt_likely", "h1b_dependent", "willful_violator", "last_fy", "score")})
        else:
            out.append({"query": n, "found": False})
    return json.dumps(out, default=str)


@tool
def semantic_search(query: str, k: int = 10) -> str:
    """Meaning-based search over employer profile cards. Use ONLY for descriptions of
    a *kind of company* that role/state filters cannot express - industry, size,
    character ('biotech startups in San Diego', 'quant trading shops', 'climate
    hardware companies').

    Do NOT use this when the query is really a role plus a constraint ('ML engineers
    at universities', 'data scientists in Boston paying over 150k') - embeddings
    match on company names there and will return firms merely *named* after the
    field. Use shortlist_sponsors for those, with cap_exempt=True for the
    university/hospital case."""
    return json.dumps(db.semantic_search(query, k=min(k, 20)), default=str)


@tool
def query_data(sql: str) -> str:
    """Run a read-only SQL SELECT against the dataset for questions the other tools
    cannot express (rankings, aggregates, trends over time).
    Tables:
      employers(employer_key, display_name, hq_city, hq_state, naics, lca_total,
        lca_certified, lca_denied, positions_total, outside_hires, renewals,
        first_seen, last_seen, last_fy, recent_filings, recent_outside,
        wage_p25, wage_median, wage_p75, h1b_dependent, willful_violator,
        perm_total, perm_certified, cap_exempt_likely, outside_ratio, score)
      employer_year(employer_key, fy_year, filings, certified, denied, positions,
        outside_hires, renewals, median_wage)
      employer_role(employer_key, soc_title, filings, outside_hires, median_wage,
        last_year, example_title)
      employer_location(employer_key, city, state, filings, outside_hires,
        median_wage, last_year)
      lca(employer_key, case_status, visa_class, decision_date, fy_year, job_title,
        soc_title, worksite_city, worksite_state, positions, new_employment,
        continued_employment, change_employer, wage_from, pw_level)
      perm(employer_key, case_status, decision_date, fy_year, job_title, soc_title,
        worksite_city, worksite_state, wage_from)
    fy_year is the US federal fiscal year; FY2026 holds Q1 (Oct-Dec 2025) only.
    IMPORTANT: employers/employer_year/employer_role/employer_location cover
    FY2023-FY2026. The per-case lca and perm tables are windowed to FY2025+.
    For anything spanning earlier years, use employer_year, not lca."""
    return json.dumps(db.run_sql(sql), default=str)


TOOLS = [lookup_company, shortlist_sponsors, compare_companies, semantic_search, query_data]
BY_NAME = {t.name: t for t in TOOLS}


# ----------------------------------------------------------------- graph
class State(TypedDict):
    messages: Annotated[list, add_messages]
    system: str


def _llm():
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=MODEL, api_key=_KEY, timeout=90)
    elif PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0,
                                     max_output_tokens=2000, google_api_key=_KEY)
    elif PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=MODEL, max_tokens=2000, temperature=0,
                            api_key=_KEY)
    else:
        raise RuntimeError("No LLM key set. Provide OPENAI_API_KEY, "
                           "GOOGLE_API_KEY or ANTHROPIC_API_KEY.")
    return llm.bind_tools(TOOLS)


def agent_node(state: State):
    sys_text = state.get("system") or SYSTEM
    return {"messages": [_llm().invoke([SystemMessage(content=sys_text)] + state["messages"])]}


def tool_node(state: State):
    last = state["messages"][-1]
    out = []
    for call in last.tool_calls:
        try:
            result = BY_NAME[call["name"]].invoke(call["args"])
        except Exception as e:
            result = json.dumps({"error": "%s: %s" % (type(e).__name__, e)})
        out.append(ToolMessage(content=str(result)[:60000], tool_call_id=call["id"]))
    return {"messages": out}


def should_continue(state: State):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def build_graph():
    g = StateGraph(State)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


_graph = None
def graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _profile_note(p):
    """Turn the saved profile into a short instruction block.

    Kept as guidance rather than a hard filter: a student asking about a specific
    company still wants the answer about that company, not a silent rewrite of
    their question into their saved preferences.
    """
    if not p:
        return ""
    bits = []
    if p.get("role"):    bits.append("target role: %s" % p["role"])
    if p.get("city") or p.get("state"):
        bits.append("target location: %s" % ", ".join(x for x in (p.get("city"), p.get("state")) if x))
    if p.get("wage"):    bits.append("minimum salary: $%s" % p["wage"])
    if p.get("degree"):  bits.append("degree: %s" % p["degree"])
    if p.get("missedcap"):
        bits.append("MISSED the H-1B cap this year - cap-exempt employers "
                    "(universities, hospitals) are especially relevant, and for "
                    "cap-subject employers say when they would need to apply")
    if p.get("nostaff"): bits.append("wants to avoid H-1B-dependent staffing firms")
    if not bits:
        return ""
    return ("\n\nThe person you are answering has this profile:\n- " + "\n- ".join(bits) +
            "\nUse it to fill in unstated filters and to tailor the recommendation. "
            "If their question names something that conflicts with the profile, the "
            "question wins.")


def ask(question, history=None, max_steps=8, profile=None):
    """Run one question through the agent. Returns answer text + tool trace."""
    sys_text = SYSTEM + _profile_note(profile)
    msgs = list(history or []) + [HumanMessage(content=question)]
    trace = []
    state = {"messages": msgs, "system": sys_text}
    for _ in range(max_steps):
        state = graph().invoke(state, config={"recursion_limit": 25})
        break
    for m in state["messages"]:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for c in m.tool_calls:
                trace.append({"tool": c["name"], "args": c["args"]})
    final = [m for m in state["messages"] if isinstance(m, AIMessage) and m.content]
    text = final[-1].content if final else "(no answer)"
    if isinstance(text, list):
        text = " ".join(b.get("text", "") for b in text if isinstance(b, dict))
    return {"answer": text, "trace": trace}
