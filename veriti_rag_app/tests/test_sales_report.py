import json

from veriti_rag import cli, reports
from veriti_rag.config import Settings
from veriti_rag.db import load_csvs
from veriti_rag.sales_report import ANALYSIS_KEYS, build_sales_report_data
from veriti_rag.sales_report_agent import (
    ENHANCE_SYSTEM,
    NARRATIVE_SYSTEM,
    PLAN_SYSTEM,
    REVIEW_SYSTEM,
    _fallback_narrative,
    run_sales_report_agent,
)


def test_sales_report_data_includes_requested_sections():
    data = build_sales_report_data(load_csvs("data/raw"))

    assert data["metrics"]["total_sales"] > 0
    assert data["metrics"]["average_sales"] > 0
    assert data["metrics"]["average_weeks_to_close"] > 0
    assert data["metrics"]["new_opportunities"] > data["metrics"]["open_opportunities"]
    assert data["metrics"]["potential_to_close"] > 0
    assert data["metrics"]["unpriced_open_opportunities"] > 0
    assert data["team_performance"]["by_manager"]
    assert data["team_performance"]["by_region"]
    assert data["agent_performance"]
    assert data["quarterly_trends"]["sales_by_close_quarter"]
    assert data["quarterly_trends"]["new_opportunities_by_engage_quarter"]
    assert data["product_performance"]
    assert data["sector_performance"]
    assert len(data["top_accounts"]) == 5
    assert all(chart["labels"] and len(chart["labels"]) == len(chart["values"]) for chart in data["charts"])


def test_sales_report_agent_filters_plan_and_repairs_coverage():
    calls = []
    responses = {
        ENHANCE_SYSTEM: {"enhanced_brief": "Fixed brief with optional conversion focus."},
        PLAN_SYSTEM: {"analysis_keys": ["teams", "unsafe_sql"], "focus": "Compare teams."},
        REVIEW_SYSTEM: {"additional_analysis_keys": ["products", "unsafe_sql"], "coverage_notes": "Review complete."},
        NARRATIVE_SYSTEM: {
            "executive_summary": "A grounded summary.",
            "key_findings": {"team_performance": "Teams compared."},
            "business_insights": {"total_sales": "Won revenue summarized."},
            "recommendations": [],
        },
    }

    def llm(system, user):
        calls.append((system, user))
        return json.dumps(responses[system])

    result = run_sales_report_agent(load_csvs("data/raw"), llm, "focus on conversion")

    assert "focus on conversion" in calls[0][1]
    assert result["planned_keys"] == list(dict.fromkeys(["teams", "products", *ANALYSIS_KEYS]))
    assert "unsafe_sql" not in result["planned_keys"]
    assert result["report_data"]["metrics"]["total_sales"] > 0
    assert set(result["narrative"]["key_findings"]) >= {
        "team_performance", "agent_performance", "quarter_trends", "product_win_rates"
    }
    assert "top_accounts" in result["narrative"]["business_insights"]
    assert len(calls) == 4


def test_sales_report_pdf_has_business_sections_without_technical_details(tmp_path, monkeypatch):
    captured = {}

    def capture_story(document, story):
        captured["text"] = "\n".join(item.getPlainText() for item in story if hasattr(item, "getPlainText"))
        captured["images"] = sum(item.__class__.__name__ == "Image" for item in story)

    monkeypatch.setattr(reports, "_build_pdf", capture_story)
    data = build_sales_report_data(load_csvs("data/raw"))
    output = reports.create_sales_report(
        {"report_data": data, "narrative": _fallback_narrative(data)},
        tmp_path / "sales_report.pdf",
    )

    assert output.name == "sales_report.pdf"
    assert "Key Findings" in captured["text"]
    assert "How is each sales team performing compared to the rest?" in captured["text"]
    assert "Sales and conversion by product" in captured["text"]
    assert "Top five accounts by won revenue" in captured["text"]
    best_rate_product = max(
        (row for row in data["product_performance"] if row["win_rate_pct"] is not None),
        key=lambda row: row["win_rate_pct"],
    )
    assert best_rate_product["product"] in captured["text"]
    assert "highest observed product win rate" in captured["text"]
    latest_quarter = data["quarterly_trends"]["sales_by_close_quarter"][-1]
    if latest_quarter["qoq_change_pct"] is not None:
        assert f"{latest_quarter['qoq_change_pct']:+.1f}%" in captured["text"]
    assert "not a probability-weighted forecast" in captured["text"]
    assert "Recommendations" in captured["text"]
    assert "SQL evidence" not in captured["text"]
    assert "Citations" not in captured["text"]
    assert "Retrieval" not in captured["text"]
    assert "trace=" not in captured["text"]
    assert captured["images"] == len(data["charts"])


def test_sales_report_cli_prints_only_success_path(monkeypatch, capsys, tmp_path):
    observed = {}
    expected_result = {"report_data": {}, "narrative": {}}
    expected_output = tmp_path / "sales.pdf"
    monkeypatch.setattr(cli, "Settings", lambda: Settings(data_dir="data/raw"))
    monkeypatch.setattr(cli, "make_llm", lambda settings: "fake-model")
    monkeypatch.setattr(cli, "make_embeddings", lambda settings: (_ for _ in ()).throw(AssertionError("embeddings must not be loaded")))
    monkeypatch.setattr(cli, "get_vectorstore", lambda *args: (_ for _ in ()).throw(AssertionError("vector store must not be loaded")))
    monkeypatch.setattr(cli, "run_sales_report_agent", lambda con, llm, prompt: observed.update(prompt=prompt, llm=llm) or expected_result)
    monkeypatch.setattr(cli, "create_sales_report", lambda result, output: expected_output)

    cli.main(["sales-report", "--output", str(expected_output), "--prompt", "emphasize conversion"])

    assert observed == {"prompt": "emphasize conversion", "llm": "fake-model"}
    assert capsys.readouterr().out.strip() == f"Sales report saved to {expected_output}"