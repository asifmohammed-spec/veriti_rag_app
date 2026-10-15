"""Bounded planning and narrative workflow for sales report generation."""
import json
import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .sales_report import ANALYSIS_KEYS, build_sales_report_data

SALES_REPORT_BRIEF = """Create an evidence-led sales performance report.

Key Findings:
1. How is each sales team performing compared to the rest?
2. Are any sales agents lagging behind?
3. Are there any quarter-over-quarter trends?
4. Do any products have better win rates?

Business Insights:
- Total sales and average sales
- Average weeks to close
- New opportunities and potential to close
- Performance by agent
- Sales and conversion by product
- Top sales by sector
- Top product by sales
- Top five accounts

Include a concise executive summary and practical recommendations. Distinguish recorded won revenue from estimated open-pipeline list price. Do not imply probability-weighted forecasting because no opportunity probability is available.
"""

ENHANCE_SYSTEM = """You refine report instructions for an analysis agent.
Preserve every required question and metric in the supplied fixed brief. Add only useful focus from the user's optional instructions. Do not invent data, calculations, or new report sections. Return JSON only: {"enhanced_brief": string}."""
PLAN_SYSTEM = """You plan a sales report using only the listed deterministic analysis blocks.
Select and order relevant analysis keys, but preserve every required report question and requested metric.
Do not write SQL or request retrieval. Return JSON only: {"analysis_keys": [string], "focus": string}."""
REVIEW_SYSTEM = """You review whether a planned sales report covers its fixed requirements using only the supplied available analysis blocks.
Return only allowed missing analysis keys in additional_analysis_keys. If coverage is complete, return an empty list. Do not request SQL, retrieval, or invented data. Respond as JSON: {"additional_analysis_keys": [string], "coverage_notes": string}."""
NARRATIVE_SYSTEM = """You write a concise business-facing sales report from deterministic computed results.
Use only figures and comparisons present in the supplied data. Do not calculate, estimate, infer causation, invent facts, mention SQL, retrieval, prompts, agent reasoning, or internal traces. Note when samples are too small and disclose data caveats. Return valid JSON only with keys: executive_summary (string), key_findings (object with team_performance, agent_performance, quarter_trends, product_win_rates strings), business_insights (object with total_sales, average_sales, average_weeks_to_close, opportunities, agent_performance, product_performance, sectors, top_product, top_accounts strings), recommendations (array of strings)."""


class SalesReportState(TypedDict, total=False):
    user_prompt: str
    enhanced_brief: str
    planned_keys: list[str]
    plan_focus: str
    report_data: dict
    review_notes: str
    narrative: dict


def _parse_json(text):
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model response did not contain a JSON object.")
    return json.loads(cleaned[start:end + 1])


def _valid_keys(values):
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(key for key in values if isinstance(key, str) and key in ANALYSIS_KEYS))


def _fallback_narrative(data):
    metrics = data["metrics"]
    manager_rows = data["team_performance"]["by_manager"]
    agent_rows = data["agent_performance"]
    quarter_rows = data["quarterly_trends"]["sales_by_close_quarter"]
    product_rows = data["product_performance"]
    sector_rows = data["sector_performance"]
    account_rows = data["top_accounts"]
    top_manager = manager_rows[0]["group_name"] if manager_rows else "No team data available"
    leading_team = manager_rows[0] if manager_rows else None
    lagging_agents = [row["sales_agent"] for row in agent_rows if row["performance"] == "below overall win rate"]
    top_product = product_rows[0]["product"] if product_rows else "No product data available"
    rate_products = [row for row in product_rows if row["win_rate_pct"] is not None]
    best_win_rate_product = max(rate_products, key=lambda row: row["win_rate_pct"]) if rate_products else None
    top_sector = sector_rows[0]["sector"] if sector_rows else "No sector data available"
    latest_quarter = quarter_rows[-1] if quarter_rows else None
    if latest_quarter and latest_quarter["qoq_change_pct"] is not None:
        quarter_text = (
            f"Won revenue in {latest_quarter['quarter']} was {latest_quarter['won_revenue']:,.2f}, "
            f"a {latest_quarter['qoq_change_pct']:+.1f}% quarter-over-quarter change."
        )
    elif latest_quarter:
        quarter_text = f"Won revenue in {latest_quarter['quarter']} was {latest_quarter['won_revenue']:,.2f}; no prior-quarter baseline is available."
    else:
        quarter_text = "Quarterly sales data is unavailable."
    agent_text = (
        "Agents below the overall win rate with at least five closed deals: " + ", ".join(lagging_agents) + "."
        if lagging_agents else "No agent with at least five closed deals is below the overall win rate."
    )
    top_accounts_text = ", ".join(row["account"] for row in account_rows) or "No account data available"
    return {
        "executive_summary": (
            f"Recorded won sales total {metrics['total_sales']:,.2f} across {metrics['won_deals']:,} won deals. "
            f"Open pipeline list-price potential is {metrics['potential_to_close']:,.2f} and is not a forecast."
        ),
        "key_findings": {
            "team_performance": (
                f"{top_manager} leads manager teams by won revenue at {leading_team['won_revenue']:,.2f}; "
                f"its win rate is {_format_pct(leading_team['win_rate_pct'])}."
                if leading_team else "Manager team performance data is unavailable."
            ),
            "agent_performance": agent_text,
            "quarter_trends": quarter_text,
            "product_win_rates": (
                f"{best_win_rate_product['product']} has the highest observed product win rate at "
                f"{best_win_rate_product['win_rate_pct']:.1f}% across {best_win_rate_product['closed_deals']:,} closed deals."
                if best_win_rate_product else "Product win rates are unavailable because no products have closed deals."
            ),
        },
        "business_insights": {
            "total_sales": f"Recorded won revenue: {metrics['total_sales']:,.2f}.",
            "average_sales": (
                f"Average won deal value: {metrics['average_sales']:,.2f}."
                if metrics["average_sales"] is not None else "Average won deal value is unavailable."
            ),
            "average_weeks_to_close": (
                f"Average time to close: {metrics['average_weeks_to_close']:.2f} weeks for valid won deals."
                if metrics["average_weeks_to_close"] is not None else "Average time to close is unavailable."
            ),
            "opportunities": (
                f"{metrics['new_opportunities']:,} opportunities; {metrics['open_opportunities']:,} open. "
                f"Open list-price potential: {metrics['potential_to_close']:,.2f}."
            ),
            "agent_performance": agent_text,
            "product_performance": f"Top product by won revenue: {top_product}.",
            "sectors": f"Top sector by won revenue: {top_sector}.",
            "top_product": f"{top_product} leads recorded won revenue by product.",
            "top_accounts": f"Top accounts by won revenue: {top_accounts_text}.",
        },
        "recommendations": [
            f"Review product matching for {metrics['unpriced_open_opportunities']:,} open opportunities without a catalog price."
        ] if metrics["unpriced_open_opportunities"] else [],
    }


def _format_pct(value):
    return f"{value:.1f}%" if value is not None else "unavailable"


def build_sales_report_agent(con, llm):
    """Build a fixed-scope agent; calculations are always delegated to Python/DuckDB."""
    def enhance(state):
        user_prompt = state.get("user_prompt", "").strip()
        request = f"FIXED REPORT BRIEF:\n{SALES_REPORT_BRIEF}\n\nOPTIONAL USER FOCUS:\n{user_prompt or '(none)'}"
        try:
            enhanced = _parse_json(llm(ENHANCE_SYSTEM, request)).get("enhanced_brief")
            if not isinstance(enhanced, str) or not enhanced.strip():
                enhanced = SALES_REPORT_BRIEF
        except Exception:
            enhanced = SALES_REPORT_BRIEF
        return {"enhanced_brief": enhanced}

    def plan(state):
        user = (
            f"REPORT BRIEF:\n{state['enhanced_brief']}\n\n"
            f"ALLOWED ANALYSIS KEYS:\n{list(ANALYSIS_KEYS)}"
        )
        try:
            result = _parse_json(llm(PLAN_SYSTEM, user))
            keys = _valid_keys(result.get("analysis_keys"))
            focus = result.get("focus", "")
        except Exception:
            keys, focus = [], ""
        return {"planned_keys": keys, "plan_focus": focus if isinstance(focus, str) else ""}

    def analyze(state):
        return {"report_data": build_sales_report_data(con)}

    def review(state):
        available = state["report_data"]["analysis_keys"]
        user = (
            f"REQUIRED REPORT BRIEF:\n{SALES_REPORT_BRIEF}\n\n"
            f"PLANNED KEYS:\n{state.get('planned_keys', [])}\n\n"
            f"AVAILABLE COMPUTED KEYS:\n{available}"
        )
        try:
            notes = _parse_json(llm(REVIEW_SYSTEM, user))
            additional = _valid_keys(notes.get("additional_analysis_keys"))
            coverage_notes = notes.get("coverage_notes", "")
        except Exception:
            additional, coverage_notes = [], ""
        missing = [key for key in ANALYSIS_KEYS if key not in state.get("planned_keys", [])]
        return {
            "planned_keys": list(dict.fromkeys(state.get("planned_keys", []) + additional + missing)),
            "review_notes": coverage_notes if isinstance(coverage_notes, str) else "",
        }

    def narrate(state):
        data = state["report_data"]
        user = json.dumps({
            "brief": state["enhanced_brief"],
            "planned_focus": state.get("plan_focus", ""),
            "review_notes": state.get("review_notes", ""),
            "metrics": data["metrics"],
            "team_performance": data["team_performance"],
            "agent_performance": data["agent_performance"],
            "quarterly_trends": data["quarterly_trends"],
            "product_performance": data["product_performance"],
            "sector_performance": data["sector_performance"],
            "top_accounts": data["top_accounts"],
            "data_quality": data["data_quality"],
        }, default=str)
        try:
            narrative = _parse_json(llm(NARRATIVE_SYSTEM, user))
            if not isinstance(narrative.get("key_findings"), dict):
                raise ValueError("Narrative is missing key findings.")
            if not isinstance(narrative.get("business_insights"), dict):
                raise ValueError("Narrative is missing business insights.")
        except Exception:
            narrative = _fallback_narrative(data)
        fallback = _fallback_narrative(data)
        for key, value in fallback.items():
            if key in {"key_findings", "business_insights"}:
                if not isinstance(narrative.get(key), dict):
                    narrative[key] = {}
                for field, field_value in value.items():
                    if not isinstance(narrative[key].get(field), str) or not narrative[key][field].strip():
                        narrative[key][field] = field_value
            elif key == "recommendations":
                if not isinstance(narrative.get(key), list):
                    narrative[key] = value
            elif not isinstance(narrative.get(key), str) or not narrative[key].strip():
                narrative[key] = value
        return {"narrative": narrative}

    workflow = StateGraph(SalesReportState)
    workflow.add_node("enhance_prompt", enhance)
    workflow.add_node("plan", plan)
    workflow.add_node("analyze", analyze)
    workflow.add_node("review", review)
    workflow.add_node("narrate", narrate)
    workflow.add_edge(START, "enhance_prompt")
    workflow.add_edge("enhance_prompt", "plan")
    workflow.add_edge("plan", "analyze")
    workflow.add_edge("analyze", "review")
    workflow.add_edge("review", "narrate")
    workflow.add_edge("narrate", END)
    return workflow.compile()


def run_sales_report_agent(con, llm, user_prompt=""):
    """Run the bounded report workflow and return data plus business narrative."""
    return build_sales_report_agent(con, llm).invoke({"user_prompt": user_prompt})