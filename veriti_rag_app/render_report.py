"""Render sales_report.template.html with JSON data; optionally export PDF."""
import argparse
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def render_report_data(data, html_path, pdf_path=None):
    """Render a template-compatible report object to HTML and optionally PDF."""
    if not isinstance(data, dict):
        raise TypeError("Report data must be a JSON object.")
    serialized = json.dumps(data, allow_nan=False)
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    environment = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=False)
    html = environment.get_template("sales_report.template.html").render(data=json.loads(serialized))
    html_path.write_text(html, encoding="utf-8")

    pdf_output = None
    if pdf_path is not None:
        pdf_output = Path(pdf_path)
        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "PDF export requires Playwright. Install it with `pip install playwright` "
                "and run `playwright install chromium`. The HTML report was still written."
            ) from exc

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(html_path.resolve().as_uri(), wait_until="load")
            page.wait_for_function("document.querySelector('#tAgent tbody') !== null")
            page.pdf(
                path=str(pdf_output),
                format="A4",
                print_background=True,
                margin={"top": "12mm", "bottom": "12mm", "left": "12mm", "right": "12mm"},
            )
            browser.close()
    return html_path, pdf_output


def render_report(data_path, out_dir, pdf=False):
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    html_path = out_dir / "report.html"
    pdf_path = out_dir / "report.pdf" if pdf else None
    return render_report_data(data, html_path, pdf_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_json", help="Template-compatible report JSON")
    parser.add_argument("out_dir", help="Directory for report.html and optional report.pdf")
    parser.add_argument("--pdf", action="store_true", help="Also export report.pdf using Playwright Chromium")
    args = parser.parse_args(argv)
    html_path, pdf_path = render_report(args.data_json, args.out_dir, args.pdf)
    print(f"HTML: {html_path}")
    if pdf_path:
        print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()