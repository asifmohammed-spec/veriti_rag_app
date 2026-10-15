from veriti_rag.db import load_csvs
from veriti_rag import cli, reports
from veriti_rag.config import Settings
from veriti_rag.reports import build_dataset_report_data, create_dataset_report, create_question_report


def test_dataset_report_data_contains_sales_insights():
    data = build_dataset_report_data(load_csvs("data/raw"))

    assert {table["name"] for table in data["tables"]} >= {"products", "sales_pipeline"}
    assert any("won" in insight.lower() for insight in data["insights"])
    assert any(chart["title"] == "Product list prices" for chart in data["charts"])
    assert any("win rate" in insight.lower() for insight in data["insights"])
    assert any("open pipeline" in insight.lower() for insight in data["insights"])


def test_dataset_report_creates_pdf(tmp_path):
    output = tmp_path / "dataset_report.pdf"

    create_dataset_report(load_csvs("data/raw"), output)

    assert output.read_bytes().startswith(b"%PDF")


def test_question_report_includes_sql_result_and_citations(tmp_path):
    output = tmp_path / "question_report.pdf"
    result = {
        "question": "Which product has the highest price?",
        "answer": "GTX Pro has the highest listed price [S1].",
        "grounded": True,
        "sql": "SELECT product, sales_price FROM products ORDER BY sales_price DESC",
        "columns": ["product", "sales_price"],
        "rows": [["GTX Pro", 4821], ["GTK 500", 26768]],
        "citations": [{"ref": "S1", "source": "products.csv"}],
    }

    create_question_report(result, output)

    assert output.read_bytes().startswith(b"%PDF")


def test_report_cli_accepts_clean_boolean_forms(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "Settings", lambda: Settings(data_dir="data/raw"))
    monkeypatch.setattr(
        cli,
        "create_dataset_report",
        lambda con, output, clean=False: calls.append((output, clean)) or output,
    )

    for arguments, expected in [([], False), (["--clean"], True), (["--clean", "yes"], True), (["--clean", "no"], False)]:
        output = tmp_path / f"report_{len(calls)}.pdf"
        cli.main(["report", *arguments, "--output", str(output)])
        assert calls[-1] == (str(output), expected)


def test_clean_report_omits_inventory_and_sql_but_keeps_charts(tmp_path, monkeypatch):
    captured = {}

    def capture_story(document, story):
        captured["text"] = "\n".join(
            item.getPlainText() for item in story if hasattr(item, "getPlainText")
        )
        captured["images"] = sum(item.__class__.__name__ == "Image" for item in story)

    monkeypatch.setattr(reports, "_build_pdf", capture_story)
    create_dataset_report(load_csvs("data/raw"), tmp_path / "clean.pdf", clean=True)

    assert "Key insights" in captured["text"]
    assert "Recommendations" in captured["text"]
    assert "Dataset inventory" not in captured["text"]
    assert "SQL evidence" not in captured["text"]
    assert captured["images"] > 0