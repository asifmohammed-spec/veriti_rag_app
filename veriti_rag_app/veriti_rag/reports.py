"""Grounded PDF reports for individual questions and complete datasets."""
from html import escape
from pathlib import Path
import tempfile


def _identifier(name):
    return '"' + str(name).replace('"', '""') + '"'


def _format_number(value):
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.2f}"


def _pdf_components():
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Image,
            KeepTogether,
            Paragraph,
            Preformatted,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise RuntimeError("PDF reports require reportlab; install requirements.txt.") from exc

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=22, leading=27, alignment=TA_LEFT, textColor=colors.HexColor("#16324F"),
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="Section", parent=styles["Heading2"], fontSize=14, leading=18,
        textColor=colors.HexColor("#176B66"), spaceBefore=14, spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="BodySmall", parent=styles["BodyText"], fontSize=9, leading=13,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="Cell", parent=styles["BodyText"], fontSize=8, leading=10,
    ))
    styles.add(ParagraphStyle(
        name="SQL", parent=styles["Code"], fontName="Courier", fontSize=7,
        leading=9, leftIndent=5, rightIndent=5, backColor=colors.HexColor("#F1F5F5"),
        borderPadding=6,
    ))
    return {
        "colors": colors,
        "letter": letter,
        "styles": styles,
        "inch": inch,
        "Image": Image,
        "KeepTogether": KeepTogether,
        "Paragraph": Paragraph,
        "Preformatted": Preformatted,
        "SimpleDocTemplate": SimpleDocTemplate,
        "Spacer": Spacer,
        "Table": Table,
        "TableStyle": TableStyle,
    }


def _styled_table(rows, components, widths=None):
    Paragraph = components["Paragraph"]
    styles = components["styles"]
    Table = components["Table"]
    TableStyle = components["TableStyle"]
    colors = components["colors"]
    body = [[Paragraph(escape(str(cell)), styles["Cell"]) for cell in row] for row in rows]
    table = Table(body, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16324F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5D5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7F7")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _draw_footer(canvas, document):
    from reportlab.lib.units import inch

    canvas.saveState()
    canvas.setStrokeColorRGB(0.82, 0.86, 0.86)
    canvas.line(0.65 * inch, 0.55 * inch, 7.85 * inch, 0.55 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColorRGB(0.35, 0.4, 0.42)
    canvas.drawString(0.65 * inch, 0.37 * inch, "VERITI | Data insights")
    canvas.drawRightString(7.85 * inch, 0.37 * inch, f"Page {document.page}")
    canvas.restoreState()


def _make_document(output, components):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = components["SimpleDocTemplate"](
        str(output), pagesize=components["letter"], rightMargin=0.65 * components["inch"],
        leftMargin=0.65 * components["inch"], topMargin=0.65 * components["inch"],
        bottomMargin=0.75 * components["inch"], title="VERITI Data Insights",
        author="VERITI RAG",
    )
    return output, document


def _render_chart(labels, values, title, x_label, y_label, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.2))
    positions = range(len(labels))
    ax.bar(positions, values, color="#176B66", width=0.72)
    ax.set_xticks(list(positions), labels, rotation=30, ha="right")
    ax.set_title(title, loc="left", fontsize=13, color="#16324F", pad=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _chart_from_result(columns, rows):
    if len(columns) != 2 or not rows:
        return None
    labels, values = [], []
    for row in rows:
        if len(row) < 2 or row[1] is None:
            return None
        try:
            value = float(row[1])
        except (TypeError, ValueError):
            return None
        labels.append(str(row[0]))
        values.append(value)
    if not labels:
        return None
    return {
        "labels": labels[:25],
        "values": values[:25],
        "title": f"{columns[1]} by {columns[0]}",
        "x_label": str(columns[0]),
        "y_label": str(columns[1]),
    }


def create_question_report(result, output):
    """Write a PDF containing a grounded answer, SQL result, chart, and citations."""
    c = _pdf_components()
    output, document = _make_document(output, c)
    Paragraph, Spacer = c["Paragraph"], c["Spacer"]
    styles = c["styles"]
    story = [
        Paragraph("Question report", styles["ReportTitle"]),
        Paragraph(escape(str(result.get("question", "Question not recorded"))), styles["BodySmall"]),
        Spacer(1, 8),
        Paragraph("Answer", styles["Section"]),
        Paragraph(escape(str(result.get("answer", "No answer returned"))).replace("\n", "<br/>"), styles["BodySmall"]),
        Paragraph(f"Grounding status: {'verified' if result.get('grounded') else 'needs review'}", styles["BodySmall"]),
    ]

    sql = result.get("sql")
    if sql:
        story.extend([Paragraph("SQL", styles["Section"]), c["Preformatted"](escape(str(sql)), styles["SQL"])])

    citations = result.get("citations") or []
    story.append(Paragraph("Sources", styles["Section"]))
    if citations:
        citation_rows = [["Ref", "Source", "Chunk"]]
        citation_rows.extend([
            [str(item.get("ref", "")), str(item.get("source", "")), str(item.get("chunk_id", ""))]
            for item in citations
        ])
        story.append(_styled_table(citation_rows, c))
    else:
        story.append(Paragraph("No knowledge-base citations were returned.", styles["BodySmall"]))

    columns = result.get("columns") or []
    rows = result.get("rows") or []
    story.append(Paragraph("Query results", styles["Section"]))
    if columns and rows:
        shown_rows = rows[:100]
        table_rows = [[str(value) for value in columns]] + [
            ["" if value is None else str(value) for value in row[:len(columns)]]
            for row in shown_rows
        ]
        story.append(_styled_table(table_rows, c))
        if len(rows) > len(shown_rows):
            story.append(Paragraph(f"Showing {len(shown_rows)} of {len(rows)} returned rows.", styles["BodySmall"]))
        chart = _chart_from_result(columns, rows)
        if chart:
            with tempfile.TemporaryDirectory() as temporary_dir:
                image_path = Path(temporary_dir) / "question_chart.png"
                _render_chart(**chart, output=image_path)
                story.extend([
                    Paragraph("Chart", styles["Section"]),
                    c["Image"](str(image_path), width=7.0 * c["inch"], height=3.7 * c["inch"]),
                ])
                if len(rows) > 25:
                    story.append(Paragraph("The chart shows the first 25 result rows.", styles["BodySmall"]))
                _build_pdf(document, story)
                return output
    else:
        story.append(Paragraph("The query returned no tabular rows.", styles["BodySmall"]))

    _build_pdf(document, story)
    return output


def _build_pdf(document, story):
    document.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)


def _run_query(con, sql, label, evidence):
    cursor = con.execute(sql)
    columns = [column[0] for column in cursor.description]
    rows = [list(row) for row in cursor.fetchall()]
    evidence.append({"label": label, "sql": sql, "columns": columns, "rows": rows})
    return columns, rows


def build_dataset_report_data(con):
    """Compute dataset summaries, supported insights, charts, and their SQL evidence."""
    evidence, tables, insights, recommendations, charts, metrics = [], [], [], [], [], []
    table_names = [row[0] for row in con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_type = 'BASE TABLE' ORDER BY table_name"
    ).fetchall()]

    for table_name in table_names:
        quoted_table = _identifier(table_name)
        column_info = con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = ? ORDER BY ordinal_position", [table_name]
        ).fetchall()
        column_names = [row[0] for row in column_info]
        count_columns = [
            f"SUM(CASE WHEN {_identifier(name)} IS NULL THEN 1 ELSE 0 END) AS _missing_{index}"
            for index, name in enumerate(column_names)
        ]
        sql = f"SELECT COUNT(*) AS row_count, {', '.join(count_columns)} FROM {quoted_table}"
        result_columns, result_rows = _run_query(con, sql, f"Missing-value profile: {table_name}", evidence)
        row_count = int(result_rows[0][0] or 0)
        missing_fields = [
            {"name": name, "count": int(result_rows[0][index + 1] or 0)}
            for index, name in enumerate(column_names) if result_rows[0][index + 1]
        ]
        missing_cells = sum(field["count"] for field in missing_fields)
        tables.append({
            "name": table_name,
            "row_count": row_count,
            "column_count": len(column_names),
            "missing_cells": missing_cells,
            "missing_fields": missing_fields,
        })
        metrics.append({"name": f"{table_name} rows", "value": row_count})
        if missing_fields:
            for field in missing_fields:
                insights.append(
                    f"{table_name}.{field['name']} has {field['count']:,} missing values "
                    f"out of {row_count:,} rows."
                )
                recommendations.append(
                    f"Review missing values in {table_name}.{field['name']} before using it "
                    "for reporting or operational decisions."
                )

    names = set(table_names)
    if "products" in names:
        product_columns = {row[0] for row in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'products'"
        ).fetchall()}
        if {"product", "sales_price"} <= product_columns:
            sql = (
                "SELECT product, TRY_CAST(sales_price AS DOUBLE) AS sales_price "
                "FROM products WHERE product IS NOT NULL AND TRY_CAST(sales_price AS DOUBLE) IS NOT NULL "
                "ORDER BY sales_price DESC LIMIT 15"
            )
            _, rows = _run_query(con, sql, "Product list prices", evidence)
            if rows:
                top_product, top_price = rows[0]
                insights.append(f"The highest listed product price is {_format_number(top_price)} for {top_product}; list price is not realized deal revenue.")
                charts.append({
                    "title": "Product list prices", "x_label": "Product", "y_label": "List price",
                    "labels": [str(row[0]) for row in rows],
                    "values": [float(row[1]) for row in rows],
                })

    if "accounts" in names:
        account_columns = {row[0] for row in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'accounts'"
        ).fetchall()}
        if "sector" in account_columns:
            sql = (
                "SELECT sector, COUNT(*) AS account_count FROM accounts "
                "WHERE sector IS NOT NULL GROUP BY sector ORDER BY account_count DESC LIMIT 10"
            )
            _, sector_rows = _run_query(con, sql, "Accounts by sector", evidence)
            if sector_rows:
                sector, count = sector_rows[0]
                insights.append(f"{sector} is the largest account sector with {_format_number(count)} accounts.")
                charts.append({
                    "title": "Accounts by sector", "x_label": "Sector", "y_label": "Accounts",
                    "labels": [str(row[0]) for row in sector_rows],
                    "values": [float(row[1]) for row in sector_rows],
                })
        if {"account", "revenue"} <= account_columns:
            sql = (
                "SELECT account, TRY_CAST(revenue AS DOUBLE) AS account_revenue FROM accounts "
                "WHERE account IS NOT NULL AND TRY_CAST(revenue AS DOUBLE) IS NOT NULL "
                "ORDER BY account_revenue DESC LIMIT 10"
            )
            _, account_rows = _run_query(con, sql, "Accounts by recorded revenue", evidence)
            if account_rows:
                account, revenue = account_rows[0]
                insights.append(f"{account} has the highest recorded account revenue at {_format_number(revenue)}.")
                charts.append({
                    "title": "Top accounts by recorded revenue", "x_label": "Account", "y_label": "Recorded revenue",
                    "labels": [str(row[0]) for row in account_rows],
                    "values": [float(row[1]) for row in account_rows],
                })

    if "sales_teams" in names:
        team_columns = {row[0] for row in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'sales_teams'"
        ).fetchall()}
        if {"sales_agent", "regional_office"} <= team_columns:
            sql = (
                "SELECT regional_office, COUNT(DISTINCT sales_agent) AS agent_count FROM sales_teams "
                "WHERE regional_office IS NOT NULL GROUP BY regional_office ORDER BY agent_count DESC"
            )
            _, region_rows = _run_query(con, sql, "Sales agents by region", evidence)
            if region_rows:
                region, count = region_rows[0]
                insights.append(f"{region} has the largest listed sales team with {_format_number(count)} agents.")
                charts.append({
                    "title": "Sales agents by region", "x_label": "Regional office", "y_label": "Sales agents",
                    "labels": [str(row[0]) for row in region_rows],
                    "values": [float(row[1]) for row in region_rows],
                })

    if "sales_pipeline" in names:
        pipeline_columns = {row[0] for row in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'sales_pipeline'"
        ).fetchall()}
        if "deal_stage" in pipeline_columns:
            sql = (
                "SELECT deal_stage, COUNT(*) AS opportunity_count FROM sales_pipeline "
                "WHERE deal_stage IS NOT NULL GROUP BY deal_stage ORDER BY opportunity_count DESC"
            )
            _, stage_rows = _run_query(con, sql, "Pipeline by deal stage", evidence)
            if stage_rows:
                insights.append(
                    f"The sales pipeline contains {sum(int(row[1]) for row in stage_rows):,} opportunities "
                    f"across {len(stage_rows)} deal stages."
                )
                charts.append({
                    "title": "Opportunities by stage", "x_label": "Deal stage", "y_label": "Opportunities",
                    "labels": [str(row[0]) for row in stage_rows],
                    "values": [float(row[1]) for row in stage_rows],
                })

        if {"deal_stage", "close_value"} <= pipeline_columns:
            sql = (
                "SELECT COUNT(*) AS won_opportunities, "
                "COALESCE(SUM(TRY_CAST(close_value AS DOUBLE)), 0) AS won_revenue "
                "FROM sales_pipeline WHERE LOWER(CAST(deal_stage AS VARCHAR)) = 'won'"
            )
            _, won_rows = _run_query(con, sql, "Won opportunity totals", evidence)
            won_count, won_revenue = won_rows[0]
            metrics.extend([
                {"name": "Won opportunities", "value": int(won_count or 0)},
                {"name": "Won revenue", "value": float(won_revenue or 0)},
            ])
            insights.append(
                f"Won opportunities total {_format_number(won_count)} records and "
                f"{_format_number(won_revenue)} in recorded close value."
            )
            if "product" in pipeline_columns:
                sql = (
                    "SELECT product, SUM(TRY_CAST(close_value AS DOUBLE)) AS won_revenue "
                    "FROM sales_pipeline WHERE LOWER(CAST(deal_stage AS VARCHAR)) = 'won' "
                    "AND TRY_CAST(close_value AS DOUBLE) IS NOT NULL AND product IS NOT NULL "
                    "GROUP BY product ORDER BY won_revenue DESC LIMIT 10"
                )
                _, product_rows = _run_query(con, sql, "Won revenue by product", evidence)
                if product_rows:
                    product, revenue = product_rows[0]
                    insights.append(
                        f"{product} leads recorded won revenue at {_format_number(revenue)}."
                    )
                    charts.append({
                        "title": "Won revenue by product", "x_label": "Product", "y_label": "Won revenue",
                        "labels": [str(row[0]) for row in product_rows],
                        "values": [float(row[1] or 0) for row in product_rows],
                    })

            if "close_value" in pipeline_columns:
                sql = (
                    "SELECT COUNT(*) AS won_missing_close_value FROM sales_pipeline "
                    "WHERE LOWER(CAST(deal_stage AS VARCHAR)) = 'won' AND close_value IS NULL"
                )
                _, missing_won_rows = _run_query(con, sql, "Won opportunities missing close value", evidence)
                missing_won = int(missing_won_rows[0][0] or 0)
                if missing_won:
                    insights.append(f"{missing_won:,} won opportunities have no close value recorded.")
                    recommendations.append(
                        f"Complete close_value for the {missing_won:,} won opportunities that are missing it; "
                        "they cannot contribute to recorded revenue totals."
                    )

        if "deal_stage" in pipeline_columns:
            sql = (
                "SELECT SUM(CASE WHEN LOWER(CAST(deal_stage AS VARCHAR)) = 'won' THEN 1 ELSE 0 END) AS won, "
                "SUM(CASE WHEN LOWER(CAST(deal_stage AS VARCHAR)) IN ('won', 'lost') THEN 1 ELSE 0 END) AS closed, "
                "100.0 * SUM(CASE WHEN LOWER(CAST(deal_stage AS VARCHAR)) = 'won' THEN 1 ELSE 0 END) "
                "/ NULLIF(SUM(CASE WHEN LOWER(CAST(deal_stage AS VARCHAR)) IN ('won', 'lost') THEN 1 ELSE 0 END), 0) "
                "AS win_rate_pct FROM sales_pipeline"
            )
            _, win_rows = _run_query(con, sql, "Win rate (won / closed)", evidence)
            won, closed, win_rate = win_rows[0]
            if closed:
                metrics.append({"name": "Win rate (%)", "value": float(win_rate)})
                insights.append(
                    f"The win rate is {_format_number(win_rate)}% based on {_format_number(won)} Won deals "
                    f"out of {_format_number(closed)} closed deals; open deals are excluded."
                )

        if {"deal_stage", "engage_date", "close_date"} <= pipeline_columns:
            sql = (
                "SELECT AVG(date_diff('day', TRY_CAST(engage_date AS DATE), TRY_CAST(close_date AS DATE))) "
                "AS average_sales_cycle_days, COUNT(*) AS valid_closed_deals "
                "FROM sales_pipeline WHERE LOWER(CAST(deal_stage AS VARCHAR)) IN ('won', 'lost') "
                "AND TRY_CAST(engage_date AS DATE) IS NOT NULL AND TRY_CAST(close_date AS DATE) IS NOT NULL "
                "AND TRY_CAST(close_date AS DATE) >= TRY_CAST(engage_date AS DATE)"
            )
            _, cycle_rows = _run_query(con, sql, "Valid closed-deal sales cycle", evidence)
            average_days, valid_deals = cycle_rows[0]
            if average_days is not None:
                metrics.append({"name": "Average sales cycle (days)", "value": float(average_days)})
                insights.append(
                    f"The average sales cycle is {_format_number(average_days)} days across "
                    f"{_format_number(valid_deals)} valid closed deals."
                )
            sql = (
                "SELECT COUNT(*) AS invalid_date_order FROM sales_pipeline "
                "WHERE LOWER(CAST(deal_stage AS VARCHAR)) IN ('won', 'lost') "
                "AND TRY_CAST(engage_date AS DATE) IS NOT NULL AND TRY_CAST(close_date AS DATE) IS NOT NULL "
                "AND TRY_CAST(close_date AS DATE) < TRY_CAST(engage_date AS DATE)"
            )
            _, invalid_rows = _run_query(con, sql, "Closed deals with invalid date order", evidence)
            invalid_dates = int(invalid_rows[0][0] or 0)
            if invalid_dates:
                insights.append(f"{invalid_dates:,} closed deals have a close date earlier than the engage date and are excluded from sales-cycle averages.")
                recommendations.append("Correct closed-deal dates where close_date is earlier than engage_date.")

        if {"deal_stage", "product"} <= pipeline_columns and "products" in names:
            product_columns = {row[0] for row in con.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'products'"
            ).fetchall()}
            if {"product", "sales_price"} <= product_columns:
                sql = (
                    "SELECT COUNT(*) AS open_deals, "
                    "SUM(COALESCE(TRY_CAST(p.sales_price AS DOUBLE), 0)) AS open_pipeline_value, "
                    "SUM(CASE WHEN p.product IS NULL OR TRY_CAST(p.sales_price AS DOUBLE) IS NULL "
                    "THEN 1 ELSE 0 END) AS unpriced_open_deals "
                    "FROM sales_pipeline AS s LEFT JOIN products AS p ON s.product = p.product "
                    "WHERE LOWER(CAST(s.deal_stage AS VARCHAR)) IN ('prospecting', 'engaging')"
                )
                _, pipeline_totals = _run_query(con, sql, "Open pipeline totals (list-price basis)", evidence)
                total_deals, total_value, unpriced_deals = pipeline_totals[0]
                sql = (
                    "SELECT s.product, COUNT(*) AS open_deals, "
                    "SUM(COALESCE(TRY_CAST(p.sales_price AS DOUBLE), 0)) AS pipeline_value "
                    "FROM sales_pipeline AS s LEFT JOIN products AS p ON s.product = p.product "
                    "WHERE LOWER(CAST(s.deal_stage AS VARCHAR)) IN ('prospecting', 'engaging') "
                    "AND s.product IS NOT NULL "
                    "GROUP BY s.product ORDER BY pipeline_value DESC LIMIT 10"
                )
                _, pipeline_rows = _run_query(con, sql, "Open pipeline value by product", evidence)
                metrics.extend([
                    {"name": "Open opportunities", "value": int(total_deals or 0)},
                    {"name": "Open pipeline value (list-price basis)", "value": float(total_value or 0)},
                ])
                insights.append(
                    f"The open pipeline contains {_format_number(total_deals)} deals with an estimated "
                    f"list-price value of {_format_number(total_value)}. This is not booked revenue."
                )
                if unpriced_deals:
                    insights.append(f"{int(unpriced_deals):,} open opportunities have no matching product list price.")
                    recommendations.append("Match open opportunity product names to the product catalog so list-price pipeline values are complete.")
                if pipeline_rows:
                    charts.append({
                        "title": "Open pipeline value by product", "x_label": "Product", "y_label": "List-price value",
                        "labels": [str(row[0]) for row in pipeline_rows],
                        "values": [float(row[2] or 0) for row in pipeline_rows],
                    })

        if {"deal_stage", "sales_agent", "close_value"} <= pipeline_columns:
            sql = (
                "SELECT sales_agent, SUM(TRY_CAST(close_value AS DOUBLE)) AS won_revenue "
                "FROM sales_pipeline WHERE LOWER(CAST(deal_stage AS VARCHAR)) = 'won' "
                "AND TRY_CAST(close_value AS DOUBLE) IS NOT NULL AND sales_agent IS NOT NULL "
                "GROUP BY sales_agent ORDER BY won_revenue DESC LIMIT 10"
            )
            _, agent_rows = _run_query(con, sql, "Won revenue by sales agent", evidence)
            if agent_rows:
                agent, revenue = agent_rows[0]
                insights.append(f"{agent} leads recorded won revenue by agent at {_format_number(revenue)}.")
                charts.append({
                    "title": "Won revenue by sales agent", "x_label": "Sales agent", "y_label": "Won revenue",
                    "labels": [str(row[0]) for row in agent_rows],
                    "values": [float(row[1] or 0) for row in agent_rows],
                })

    if not insights:
        insights.append("The available tables were inspected, but no supported business insight could be derived from their columns.")
    if not recommendations:
        recommendations.append("No missing values were detected in the inspected tables; validate business-specific rules before operational use.")

    return {
        "tables": tables,
        "metrics": metrics,
        "insights": insights,
        "recommendations": recommendations,
        "charts": charts,
        "evidence": evidence,
    }


def create_dataset_report(con, output, clean=False):
    """Create a full dataset PDF, optionally omitting technical details."""
    data = build_dataset_report_data(con)
    c = _pdf_components()
    output, document = _make_document(output, c)
    Paragraph, Spacer = c["Paragraph"], c["Spacer"]
    styles = c["styles"]
    title = "Business insights" if clean else "Dataset insights"
    story = [Paragraph(title, styles["ReportTitle"])]
    if not clean:
        story.extend([
            Paragraph("Evidence-led overview of the available CRM tables.", styles["BodySmall"]),
            Paragraph("Executive summary", styles["Section"]),
        ])
        total_rows = sum(table["row_count"] for table in data["tables"])
        total_missing = sum(table["missing_cells"] for table in data["tables"])
        story.append(Paragraph(
            f"The report covers {len(data['tables'])} tables and {total_rows:,} records. "
            f"The inspected tables contain {total_missing:,} missing cells. "
            "Business findings below are computed from the source tables using DuckDB.",
            styles["BodySmall"],
        ))

        story.append(Paragraph("Dataset inventory", styles["Section"]))
        table_rows = [["Table", "Rows", "Columns", "Missing cells"]]
        table_rows.extend([
            [table["name"], _format_number(table["row_count"]), _format_number(table["column_count"]), _format_number(table["missing_cells"])]
            for table in data["tables"]
        ])
        if len(table_rows) > 1:
            story.append(_styled_table(table_rows, c))
        story.append(Paragraph("Data sources", styles["Section"]))
        source_files = ", ".join(f"{table['name']}.csv" for table in data["tables"])
        story.append(Paragraph(escape(source_files) if source_files else "No CSV tables were loaded.", styles["BodySmall"]))

        if data["metrics"]:
            story.append(Paragraph("Key metrics", styles["Section"]))
            metric_rows = [["Metric", "Value"]]
            metric_rows.extend([[metric["name"], _format_number(metric["value"])] for metric in data["metrics"]])
            story.append(_styled_table(metric_rows, c))

    story.append(Paragraph("Key insights", styles["Section"]))
    for insight in data["insights"]:
        story.append(Paragraph(f"&#8226; {escape(insight)}", styles["BodySmall"]))

    story.append(Paragraph("Recommendations", styles["Section"]))
    for recommendation in data["recommendations"]:
        story.append(Paragraph(f"&#8226; {escape(recommendation)}", styles["BodySmall"]))

    with tempfile.TemporaryDirectory() as temporary_dir:
        for index, chart in enumerate(data["charts"]):
            image_path = Path(temporary_dir) / f"chart_{index}.png"
            _render_chart(**chart, output=image_path)
            story.extend([
                Paragraph(escape(chart["title"]), styles["Section"]),
                c["Image"](str(image_path), width=7.0 * c["inch"], height=3.7 * c["inch"]),
                Spacer(1, 6),
            ])

        if not clean:
            story.append(Paragraph("SQL evidence", styles["Section"]))
            story.append(Paragraph(
                "These are the read-only queries used to compute the report findings. Source tables are loaded from the configured CSV directory.",
                styles["BodySmall"],
            ))
            for item in data["evidence"]:
                story.append(c["KeepTogether"]([
                    Paragraph(escape(item["label"]), styles["Heading3"]),
                    c["Preformatted"](escape(item["sql"]), styles["SQL"]),
                    Spacer(1, 6),
                ]))
        _build_pdf(document, story)
    return output


def _sales_value(value, kind="number"):
    if value is None:
        return "N/A"
    if kind == "percent":
        return f"{float(value):.1f}%"
    if kind == "money":
        return f"${_format_number(value)}"
    return _format_number(value)


def _sales_table(title, rows, columns, components, story):
    story.append(components["Paragraph"](escape(title), components["styles"]["Section"]))
    if not rows:
        story.append(components["Paragraph"]("No data available for this section.", components["styles"]["BodySmall"]))
        return
    table_rows = [[label for label, _, _ in columns]]
    for row in rows:
        table_rows.append([
            _sales_value(row.get(key), kind) if kind else ("N/A" if row.get(key) is None else str(row.get(key)))
            for _, key, kind in columns
        ])
    story.append(_styled_table(table_rows, components))


def create_sales_report(result, output):
    """Write a business-focused sales PDF without SQL, retrieval, or agent traces."""
    data = result["report_data"]
    narrative = result.get("narrative") or {}
    c = _pdf_components()
    output, document = _make_document(output, c)
    Paragraph, Spacer = c["Paragraph"], c["Spacer"]
    styles = c["styles"]
    story = [Paragraph("Sales performance report", styles["ReportTitle"])]
    story.extend([
        Paragraph("Executive summary", styles["Section"]),
        Paragraph(escape(str(narrative.get("executive_summary", "Sales performance summary is unavailable."))), styles["BodySmall"]),
    ])

    metrics = data["metrics"]
    metric_rows = [
        ["Total sales", _sales_value(metrics.get("total_sales"), "money")],
        ["Average sales per won deal", _sales_value(metrics.get("average_sales"), "money")],
        ["Average weeks to close", _sales_value(metrics.get("average_weeks_to_close"))],
        ["New opportunities", _sales_value(metrics.get("new_opportunities"))],
        ["Open opportunities", _sales_value(metrics.get("open_opportunities"))],
        ["Potential to close (list-price estimate)", _sales_value(metrics.get("potential_to_close"), "money")],
        ["Overall win rate", _sales_value(metrics.get("win_rate_pct"), "percent")],
    ]
    story.extend([
        Paragraph("Business insights", styles["Section"]),
        _styled_table([["Metric", "Result"], *metric_rows], c),
        Paragraph(
            "Potential to close is estimated from exact product-catalog list-price matches for open opportunities. "
            "It is not booked revenue and is not a probability-weighted forecast.",
            styles["BodySmall"],
        ),
    ])

    key_findings = narrative.get("key_findings") or {}
    story.append(Paragraph("Key Findings", styles["Section"]))
    for heading, key in [
        ("How is each sales team performing compared to the rest?", "team_performance"),
        ("Are any sales agents lagging behind?", "agent_performance"),
        ("Are there any quarter-over-quarter trends?", "quarter_trends"),
        ("Do any products have better win rates?", "product_win_rates"),
    ]:
        story.append(Paragraph(escape(heading), styles["Heading3"]))
        story.append(Paragraph(escape(str(key_findings.get(key, "See the detailed results below."))), styles["BodySmall"]))

    recommendations = narrative.get("recommendations") or []
    if isinstance(recommendations, list) and recommendations:
        story.append(Paragraph("Recommendations", styles["Section"]))
        for recommendation in recommendations:
            story.append(Paragraph(f"&#8226; {escape(str(recommendation))}", styles["BodySmall"]))

    insights = narrative.get("business_insights") or {}
    for key, heading in [
        ("total_sales", "Sales"),
        ("average_sales", "Average sales"),
        ("average_weeks_to_close", "Sales cycle"),
        ("opportunities", "Opportunities and pipeline"),
        ("agent_performance", "Agent performance"),
        ("product_performance", "Product performance"),
        ("sectors", "Sales by sector"),
        ("top_product", "Top product by sales"),
        ("top_accounts", "Top five accounts"),
    ]:
        text = insights.get(key)
        if text:
            story.append(Paragraph(escape(heading), styles["Heading3"]))
            story.append(Paragraph(escape(str(text)), styles["BodySmall"]))

    team_columns = [
        ("Manager team", "group_name", None),
        ("Opportunities", "opportunities", "number"),
        ("Won deals", "won_deals", "number"),
        ("Won revenue", "won_revenue", "money"),
        ("Win rate", "win_rate_pct", "percent"),
        ("Vs overall (pp)", "vs_overall_win_rate_pp", "number"),
    ]
    _sales_table("Performance by manager team", data["team_performance"]["by_manager"], team_columns, c, story)

    region_columns = [
        ("Region", "group_name", None),
        ("Opportunities", "opportunities", "number"),
        ("Won revenue", "won_revenue", "money"),
        ("Win rate", "win_rate_pct", "percent"),
    ]
    _sales_table("Performance by region", data["team_performance"]["by_region"], region_columns, c, story)

    agent_columns = [
        ("Agent", "sales_agent", None),
        ("Manager", "manager", None),
        ("Closed deals", "closed_deals", "number"),
        ("Won revenue", "won_revenue", "money"),
        ("Win rate", "win_rate_pct", "percent"),
        ("Comparison", "performance", None),
    ]
    _sales_table("Performance by agent", data["agent_performance"], agent_columns, c, story)
    story.append(Paragraph(
        "Agents with fewer than five closed deals are marked as having an insufficient sample. "
        "Other agent comparisons use the overall win rate.", styles["BodySmall"]
    ))

    quarter_columns = [
        ("Close quarter", "quarter", None),
        ("Won deals", "won_deals", "number"),
        ("Closed deals", "closed_deals", "number"),
        ("Won revenue", "won_revenue", "money"),
        ("Win rate", "win_rate_pct", "percent"),
        ("QoQ revenue change", "qoq_change_pct", "percent"),
    ]
    _sales_table("Quarter-over-quarter sales", data["quarterly_trends"]["sales_by_close_quarter"], quarter_columns, c, story)
    _sales_table(
        "New opportunities by engage quarter",
        data["quarterly_trends"]["new_opportunities_by_engage_quarter"],
        [("Engage quarter", "quarter", None), ("New opportunities", "new_opportunities", "number"),
         ("QoQ change", "qoq_change_pct", "percent")],
        c,
        story,
    )

    product_columns = [
        ("Product", "product", None),
        ("Opportunities", "opportunities", "number"),
        ("Closed deals", "closed_deals", "number"),
        ("Won revenue", "won_revenue", "money"),
        ("Win rate", "win_rate_pct", "percent"),
    ]
    _sales_table("Sales and conversion by product", data["product_performance"], product_columns, c, story)

    sector_columns = [
        ("Sector", "sector", None),
        ("Opportunities", "opportunities", "number"),
        ("Won deals", "won_deals", "number"),
        ("Won revenue", "won_revenue", "money"),
        ("Win rate", "win_rate_pct", "percent"),
    ]
    _sales_table("Top sectors by won revenue", data["sector_performance"], sector_columns, c, story)

    account_columns = [
        ("Account", "account", None),
        ("Sector", "sector", None),
        ("Opportunities", "opportunities", "number"),
        ("Won deals", "won_deals", "number"),
        ("Won revenue", "won_revenue", "money"),
    ]
    _sales_table("Top five accounts by won revenue", data["top_accounts"], account_columns, c, story)

    quality = data["data_quality"]
    quality_notes = []
    if quality["unpriced_open_opportunities"]:
        quality_notes.append(
            f"{quality['unpriced_open_opportunities']:,} open opportunities had no exact product list-price match "
            "and are excluded from the potential-to-close estimate."
        )
    if quality["won_missing_close_value"]:
        quality_notes.append(
            f"{quality['won_missing_close_value']:,} won deals have no recorded close value and are excluded from sales totals."
        )
    if quality["reversed_dates"]:
        quality_notes.append(
            f"{quality['reversed_dates']:,} closed deals have a close date before the engage date and are excluded from sales-cycle averages."
        )
    if quality_notes:
        story.append(Paragraph("Data notes", styles["Section"]))
        for note in quality_notes:
            story.append(Paragraph(f"&#8226; {escape(note)}", styles["BodySmall"]))

    with tempfile.TemporaryDirectory() as temporary_dir:
        for index, chart in enumerate(data["charts"]):
            image_path = Path(temporary_dir) / f"sales_chart_{index}.png"
            _render_chart(**chart, output=image_path)
            story.extend([
                Paragraph(escape(chart["title"]), styles["Section"]),
                c["Image"](str(image_path), width=7.0 * c["inch"], height=3.7 * c["inch"]),
                Spacer(1, 6),
            ])
        _build_pdf(document, story)
    return output