"""Deterministic DuckDB analyses for the business sales report."""

ANALYSIS_KEYS = (
    "overview",
    "teams",
    "agents",
    "quarters",
    "products",
    "sectors",
    "accounts",
    "data_quality",
)


def _rows(con, sql):
    cursor = con.execute(sql)
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _number(value):
    return float(value) if value is not None else None


def _integer(value):
    return int(value or 0)


def _quarter_expression(date_expression):
    return (
        f"CAST(EXTRACT(YEAR FROM {date_expression}) AS INTEGER)::VARCHAR || '-Q' || "
        f"CAST(EXTRACT(QUARTER FROM {date_expression}) AS INTEGER)::VARCHAR"
    )


def _overview(con):
    pipeline = _rows(con, """
        SELECT
            COUNT(*) AS new_opportunities,
            COUNT(*) FILTER (WHERE LOWER(deal_stage) IN ('prospecting', 'engaging')) AS open_opportunities,
            COUNT(*) FILTER (WHERE LOWER(deal_stage) = 'won') AS won_deals,
            COUNT(*) FILTER (WHERE LOWER(deal_stage) IN ('won', 'lost')) AS closed_deals,
            SUM(TRY_CAST(close_value AS DOUBLE)) FILTER (WHERE LOWER(deal_stage) = 'won') AS total_sales,
            AVG(TRY_CAST(close_value AS DOUBLE)) FILTER (
                WHERE LOWER(deal_stage) = 'won' AND TRY_CAST(close_value AS DOUBLE) IS NOT NULL
            ) AS average_sales,
            100.0 * COUNT(*) FILTER (WHERE LOWER(deal_stage) = 'won') /
                NULLIF(COUNT(*) FILTER (WHERE LOWER(deal_stage) IN ('won', 'lost')), 0) AS win_rate_pct,
            AVG(date_diff('day', TRY_CAST(engage_date AS DATE), TRY_CAST(close_date AS DATE)) / 7.0)
                FILTER (
                    WHERE LOWER(deal_stage) = 'won'
                      AND TRY_CAST(engage_date AS DATE) IS NOT NULL
                      AND TRY_CAST(close_date AS DATE) >= TRY_CAST(engage_date AS DATE)
                ) AS average_weeks_to_close
        FROM sales_pipeline
    """)[0]

    open_pipeline = _rows(con, """
        SELECT
            COUNT(*) AS open_opportunities,
            SUM(TRY_CAST(p.sales_price AS DOUBLE)) AS potential_to_close,
            COUNT(*) FILTER (WHERE TRY_CAST(p.sales_price AS DOUBLE) IS NULL) AS unpriced_open_opportunities
        FROM sales_pipeline AS s
        LEFT JOIN products AS p ON LOWER(TRIM(s.product)) = LOWER(TRIM(p.product))
        WHERE LOWER(s.deal_stage) IN ('prospecting', 'engaging')
    """)[0]

    return {
        "total_sales": _number(pipeline["total_sales"]) or 0.0,
        "average_sales": _number(pipeline["average_sales"]),
        "average_weeks_to_close": _number(pipeline["average_weeks_to_close"]),
        "new_opportunities": _integer(pipeline["new_opportunities"]),
        "open_opportunities": _integer(open_pipeline["open_opportunities"]),
        "potential_to_close": _number(open_pipeline["potential_to_close"]) or 0.0,
        "unpriced_open_opportunities": _integer(open_pipeline["unpriced_open_opportunities"]),
        "won_deals": _integer(pipeline["won_deals"]),
        "closed_deals": _integer(pipeline["closed_deals"]),
        "win_rate_pct": _number(pipeline["win_rate_pct"]),
    }


def _group_performance(con, group_expression):
    team_join = (
        "LEFT JOIN sales_teams AS t ON LOWER(TRIM(s.sales_agent)) = LOWER(TRIM(t.sales_agent))"
        if "t." in group_expression else ""
    )
    return _rows(con, f"""
        SELECT
            {group_expression} AS group_name,
            COUNT(*) AS opportunities,
            COUNT(*) FILTER (WHERE LOWER(s.deal_stage) = 'won') AS won_deals,
            COUNT(*) FILTER (WHERE LOWER(s.deal_stage) IN ('won', 'lost')) AS closed_deals,
            SUM(TRY_CAST(s.close_value AS DOUBLE)) FILTER (
                WHERE LOWER(s.deal_stage) = 'won'
            ) AS won_revenue,
            100.0 * COUNT(*) FILTER (WHERE LOWER(s.deal_stage) = 'won') /
                NULLIF(COUNT(*) FILTER (WHERE LOWER(s.deal_stage) IN ('won', 'lost')), 0) AS win_rate_pct
        FROM sales_pipeline AS s
        {team_join}
        WHERE s.sales_agent IS NOT NULL
        GROUP BY group_name
        ORDER BY won_revenue DESC NULLS LAST
    """)


def _teams(con, overall_win_rate):
    managers = _group_performance(con, "COALESCE(t.manager, 'Unassigned manager')")
    regions = _group_performance(con, "COALESCE(t.regional_office, 'Unassigned region')")
    for row in managers + regions:
        row["opportunities"] = _integer(row["opportunities"])
        row["won_deals"] = _integer(row["won_deals"])
        row["closed_deals"] = _integer(row["closed_deals"])
        row["won_revenue"] = _number(row["won_revenue"]) or 0.0
        row["win_rate_pct"] = _number(row["win_rate_pct"])
        row["vs_overall_win_rate_pp"] = (
            row["win_rate_pct"] - overall_win_rate
            if row["win_rate_pct"] is not None and overall_win_rate is not None else None
        )
    return {"by_manager": managers, "by_region": regions}


def _agents(con, overall_win_rate):
    rows = _rows(con, """
        SELECT
            s.sales_agent,
            COALESCE(t.manager, 'Unassigned manager') AS manager,
            COUNT(*) AS opportunities,
            COUNT(*) FILTER (WHERE LOWER(s.deal_stage) = 'won') AS won_deals,
            COUNT(*) FILTER (WHERE LOWER(s.deal_stage) IN ('won', 'lost')) AS closed_deals,
            SUM(TRY_CAST(s.close_value AS DOUBLE)) FILTER (
                WHERE LOWER(s.deal_stage) = 'won'
            ) AS won_revenue,
            100.0 * COUNT(*) FILTER (WHERE LOWER(s.deal_stage) = 'won') /
                NULLIF(COUNT(*) FILTER (WHERE LOWER(s.deal_stage) IN ('won', 'lost')), 0) AS win_rate_pct
        FROM sales_pipeline AS s
        LEFT JOIN sales_teams AS t ON LOWER(TRIM(s.sales_agent)) = LOWER(TRIM(t.sales_agent))
        WHERE s.sales_agent IS NOT NULL
        GROUP BY s.sales_agent, manager
        ORDER BY won_revenue DESC NULLS LAST
    """)
    for row in rows:
        row["opportunities"] = _integer(row["opportunities"])
        row["won_deals"] = _integer(row["won_deals"])
        row["closed_deals"] = _integer(row["closed_deals"])
        row["won_revenue"] = _number(row["won_revenue"]) or 0.0
        row["win_rate_pct"] = _number(row["win_rate_pct"])
        if row["closed_deals"] < 5:
            row["performance"] = "insufficient sample"
        elif overall_win_rate is not None and row["win_rate_pct"] < overall_win_rate:
            row["performance"] = "below overall win rate"
        else:
            row["performance"] = "at or above overall win rate"
    return rows


def _quarters(con):
    close_quarter = _quarter_expression("TRY_CAST(close_date AS DATE)")
    sales = _rows(con, f"""
        SELECT
            {close_quarter} AS quarter,
            COUNT(*) FILTER (WHERE LOWER(deal_stage) = 'won') AS won_deals,
            COUNT(*) FILTER (WHERE LOWER(deal_stage) IN ('won', 'lost')) AS closed_deals,
            SUM(TRY_CAST(close_value AS DOUBLE)) FILTER (
                WHERE LOWER(deal_stage) = 'won'
            ) AS won_revenue,
            100.0 * COUNT(*) FILTER (WHERE LOWER(deal_stage) = 'won') /
                NULLIF(COUNT(*) FILTER (WHERE LOWER(deal_stage) IN ('won', 'lost')), 0) AS win_rate_pct
        FROM sales_pipeline
        WHERE TRY_CAST(close_date AS DATE) IS NOT NULL
        GROUP BY quarter
        ORDER BY quarter
    """)
    engage_quarter = _quarter_expression("TRY_CAST(engage_date AS DATE)")
    new_opportunities = _rows(con, f"""
        SELECT {engage_quarter} AS quarter, COUNT(*) AS new_opportunities
        FROM sales_pipeline
        WHERE TRY_CAST(engage_date AS DATE) IS NOT NULL
        GROUP BY quarter
        ORDER BY quarter
    """)
    for row in sales:
        row["won_deals"] = _integer(row["won_deals"])
        row["closed_deals"] = _integer(row["closed_deals"])
        row["won_revenue"] = _number(row["won_revenue"]) or 0.0
        row["win_rate_pct"] = _number(row["win_rate_pct"])
    for row in new_opportunities:
        row["new_opportunities"] = _integer(row["new_opportunities"])
    for rows, key in ((sales, "won_revenue"), (new_opportunities, "new_opportunities")):
        for index, row in enumerate(rows):
            previous = rows[index - 1][key] if index else None
            row["qoq_change_pct"] = (
                100.0 * (row[key] - previous) / abs(previous)
                if previous not in (None, 0) else None
            )
    return {"sales_by_close_quarter": sales, "new_opportunities_by_engage_quarter": new_opportunities}


def _products(con):
    rows = _rows(con, """
        SELECT
            product,
            COUNT(*) AS opportunities,
            COUNT(*) FILTER (WHERE LOWER(deal_stage) = 'won') AS won_deals,
            COUNT(*) FILTER (WHERE LOWER(deal_stage) IN ('won', 'lost')) AS closed_deals,
            SUM(TRY_CAST(close_value AS DOUBLE)) FILTER (
                WHERE LOWER(deal_stage) = 'won'
            ) AS won_revenue,
            100.0 * COUNT(*) FILTER (WHERE LOWER(deal_stage) = 'won') /
                NULLIF(COUNT(*) FILTER (WHERE LOWER(deal_stage) IN ('won', 'lost')), 0) AS win_rate_pct
        FROM sales_pipeline
        WHERE product IS NOT NULL
        GROUP BY product
        ORDER BY won_revenue DESC NULLS LAST
    """)
    for row in rows:
        for key in ("opportunities", "won_deals", "closed_deals"):
            row[key] = _integer(row[key])
        row["won_revenue"] = _number(row["won_revenue"]) or 0.0
        row["win_rate_pct"] = _number(row["win_rate_pct"])
    return rows


def _sectors(con):
    rows = _rows(con, """
        SELECT
            a.sector,
            COUNT(*) AS opportunities,
            COUNT(*) FILTER (WHERE LOWER(s.deal_stage) = 'won') AS won_deals,
            COUNT(*) FILTER (WHERE LOWER(s.deal_stage) IN ('won', 'lost')) AS closed_deals,
            SUM(TRY_CAST(s.close_value AS DOUBLE)) FILTER (
                WHERE LOWER(s.deal_stage) = 'won'
            ) AS won_revenue,
            100.0 * COUNT(*) FILTER (WHERE LOWER(s.deal_stage) = 'won') /
                NULLIF(COUNT(*) FILTER (WHERE LOWER(s.deal_stage) IN ('won', 'lost')), 0) AS win_rate_pct
        FROM sales_pipeline AS s
        JOIN accounts AS a ON LOWER(TRIM(s.account)) = LOWER(TRIM(a.account))
        WHERE a.sector IS NOT NULL
        GROUP BY a.sector
        ORDER BY won_revenue DESC NULLS LAST
    """)
    for row in rows:
        for key in ("opportunities", "won_deals", "closed_deals"):
            row[key] = _integer(row[key])
        row["won_revenue"] = _number(row["won_revenue"]) or 0.0
        row["win_rate_pct"] = _number(row["win_rate_pct"])
    return rows


def _accounts(con):
    rows = _rows(con, """
        SELECT
            s.account,
            a.sector,
            COUNT(*) AS opportunities,
            COUNT(*) FILTER (WHERE LOWER(s.deal_stage) = 'won') AS won_deals,
            SUM(TRY_CAST(s.close_value AS DOUBLE)) FILTER (
                WHERE LOWER(s.deal_stage) = 'won'
            ) AS won_revenue
        FROM sales_pipeline AS s
        LEFT JOIN accounts AS a ON LOWER(TRIM(s.account)) = LOWER(TRIM(a.account))
        WHERE s.account IS NOT NULL
        GROUP BY s.account, a.sector
        ORDER BY won_revenue DESC NULLS LAST
        LIMIT 5
    """)
    for row in rows:
        row["opportunities"] = _integer(row["opportunities"])
        row["won_deals"] = _integer(row["won_deals"])
        row["won_revenue"] = _number(row["won_revenue"]) or 0.0
    return rows


def _data_quality(con, overview):
    date_rows = _rows(con, """
        SELECT
            COUNT(*) FILTER (WHERE TRY_CAST(engage_date AS DATE) IS NULL) AS invalid_engage_dates,
            COUNT(*) FILTER (WHERE TRY_CAST(close_date AS DATE) IS NULL AND LOWER(deal_stage) IN ('won', 'lost')) AS invalid_closed_dates,
            COUNT(*) FILTER (WHERE LOWER(deal_stage) = 'won' AND close_value IS NULL) AS won_missing_close_value,
            COUNT(*) FILTER (WHERE LOWER(deal_stage) IN ('won', 'lost')
                AND TRY_CAST(engage_date AS DATE) IS NOT NULL
                AND TRY_CAST(close_date AS DATE) IS NOT NULL
                AND TRY_CAST(close_date AS DATE) < TRY_CAST(engage_date AS DATE)) AS reversed_dates
        FROM sales_pipeline
    """)[0]
    return {
        **{key: _integer(value) for key, value in date_rows.items()},
        "unpriced_open_opportunities": overview["unpriced_open_opportunities"],
    }


def build_sales_report_data(con):
    """Return deterministic sales metrics, comparisons, trends, and chart series."""
    overview = _overview(con)
    teams = _teams(con, overview["win_rate_pct"])
    agents = _agents(con, overview["win_rate_pct"])
    quarters = _quarters(con)
    products = _products(con)
    sectors = _sectors(con)
    accounts = _accounts(con)
    quality = _data_quality(con, overview)

    charts = [
        {"title": "Won revenue by manager team", "x_label": "Manager", "y_label": "Won revenue",
         "labels": [row["group_name"] for row in teams["by_manager"][:12]],
         "values": [row["won_revenue"] for row in teams["by_manager"][:12]]},
        {"title": "Won revenue by sales agent", "x_label": "Sales agent", "y_label": "Won revenue",
         "labels": [row["sales_agent"] for row in agents[:12]],
         "values": [row["won_revenue"] for row in agents[:12]]},
        {"title": "Won revenue by product", "x_label": "Product", "y_label": "Won revenue",
         "labels": [row["product"] for row in products[:10]],
         "values": [row["won_revenue"] for row in products[:10]]},
        {"title": "Product win rate", "x_label": "Product", "y_label": "Win rate (%)",
         "labels": [row["product"] for row in products if row["win_rate_pct"] is not None][:10],
         "values": [row["win_rate_pct"] for row in products if row["win_rate_pct"] is not None][:10]},
        {"title": "Won revenue by sector", "x_label": "Sector", "y_label": "Won revenue",
         "labels": [row["sector"] for row in sectors[:10]],
         "values": [row["won_revenue"] for row in sectors[:10]]},
        {"title": "Top accounts by won revenue", "x_label": "Account", "y_label": "Won revenue",
         "labels": [row["account"] for row in accounts],
         "values": [row["won_revenue"] for row in accounts]},
        {"title": "Won revenue by close quarter", "x_label": "Quarter", "y_label": "Won revenue",
         "labels": [row["quarter"] for row in quarters["sales_by_close_quarter"]],
         "values": [row["won_revenue"] for row in quarters["sales_by_close_quarter"]]},
    ]
    charts = [chart for chart in charts if chart["labels"]]

    return {
        "metrics": overview,
        "team_performance": teams,
        "agent_performance": agents,
        "quarterly_trends": quarters,
        "product_performance": products,
        "sector_performance": sectors,
        "top_accounts": accounts,
        "data_quality": quality,
        "charts": charts,
        "analysis_keys": list(ANALYSIS_KEYS),
    }