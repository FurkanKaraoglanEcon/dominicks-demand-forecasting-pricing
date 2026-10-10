"""Create the economic narrative and publication-style static figures from computed outputs."""

import json
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)

NAMES = ["Pepsi", "Diet Pepsi", "Coca-Cola Classic", "Diet Coke"]
BLUE = "#173b58"
GREEN = "#168078"
ORANGE = "#c66b2b"


def markdown_table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def make_figures(output, reports):
    directory = reports / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": BLUE,
            "text.color": BLUE,
            "axes.titleweight": "bold",
        }
    )
    coverage = pd.read_csv(output / "revenue_coverage.csv")
    fig, ax = plt.subplots(figsize=(7.8, 3))
    x = np.arange(3)
    ax.bar(
        x - 0.16,
        100 * coverage.market_coverage,
        width=0.32,
        label="Broader 2L cola",
        color=GREEN,
    )
    ax.bar(
        x + 0.16,
        100 * coverage.full_category_coverage,
        width=0.32,
        label="All soft drinks",
        color=BLUE,
    )
    ax.set(
        xticks=x,
        xticklabels=["Training", "Validation", "Later period"],
        ylim=(0, 100),
        ylabel="Selected revenue / market revenue (%)",
    )
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(directory / "market_coverage.png", dpi=180)
    plt.close(fig)
    dynamics = pd.read_csv(output / "demand/demand_dynamics.csv")
    own = dynamics.loc[
        dynamics.system.eq("linked_product")
        & dynamics.price.eq((dynamics.demand > 2).astype(int) + 1)
    ]
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    for offset, response, color in [
        (-0.18, "immediate", BLUE),
        (0, "subsequent", ORANGE),
        (0.18, "cumulative", GREEN),
    ]:
        frame = own.loc[own.response.eq(response)].sort_values("demand")
        ax.errorbar(
            x=np.arange(4) + offset,
            y=frame.elasticity,
            yerr=[frame.elasticity - frame.lower95, frame.upper95 - frame.elasticity],
            fmt="o",
            capsize=3,
            label=response,
            color=color,
        )
    ax.axhline(0, color="#aaaaaa", lw=0.8)
    ax.set(
        xticks=np.arange(4),
        xticklabels=["Pepsi", "Diet Pepsi", "Coke", "Diet Coke"],
        ylabel="Brand-price response (log points)",
    )
    ax.legend(frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(directory / "demand_dynamics.png", dpi=180)
    plt.close(fig)
    scores = pd.read_csv(output / "forecast/forecast_metrics_extended.csv")
    selected = scores.loc[
        scores["sample"].eq("retrospective_test") & scores.promotion.eq(-1)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for offset, model, label, color in [
        (-0.18, "rolling_no_trend", "Rolling", BLUE),
        (0, "rolling_calibrated", "Pooled calibration", ORANGE),
        (0.18, "rolling_promo_calibrated", "Promotion calibration", GREEN),
    ]:
        frame = selected.loc[selected.model.eq(model)].sort_values("product")
        for ax, column in zip(axes, ["WAPE", "bias_pct"]):
            ax.bar(
                np.arange(4) + offset,
                frame[column],
                width=0.18,
                label=label,
                color=color,
            )
    for ax in axes:
        ax.set(
            xticks=np.arange(4),
            xticklabels=["Pepsi", "Diet\nPepsi", "Coke", "Diet\nCoke"],
        )
        ax.axhline(0, color="#aaaaaa", lw=0.7)
    axes[0].set_ylabel("WAPE (%)")
    axes[1].set_ylabel("Aggregate bias (%)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(directory / "forecast_calibration.png", dpi=180)
    plt.close(fig)
    weekly = pd.read_csv(output / "forecast/forecast_weekly_metrics.csv")
    choices = pd.read_csv(output / "forecast/forecast_choices_extended.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for j, ax in enumerate(axes.ravel(), 1):
        chosen = choices.loc[choices["product"].eq(j), "model"].iloc[0]
        frame = weekly.loc[weekly["product"].eq(j) & weekly.model.eq(chosen)]
        ax.plot(frame.week, frame.actual, color=BLUE, label="Actual")
        ax.plot(frame.week, frame.predicted, color=GREEN, label="Selected forecast")
        ax.axvline(339.5, color=ORANGE, ls="--", lw=0.8)
        ax.set_title(NAMES[j - 1])
        ax.set_ylabel("Weekly units")
        ax.set_xlabel("Database week")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(directory / "selected_forecasts.png", dpi=180)
    plt.close(fig)


def write_report(config, verification):
    output, reports = config["paths"]["output"], config["paths"]["reports"]
    reports.mkdir(parents=True, exist_ok=True)
    categories = pd.read_csv(output / "categories/category_summary.csv")
    audit = json.loads((output / "preparation_audit.json").read_text())
    coverage = pd.read_csv(output / "revenue_coverage.csv")
    prices = pd.read_csv(output / "identification/independent_price_variation.csv")
    elasticities = pd.read_csv(output / "demand/demand_sensitivity.csv")
    dynamics = pd.read_csv(output / "demand/demand_dynamics.csv")
    chosen = pd.read_csv(output / "forecast/forecast_choices_extended.csv")
    metrics = pd.read_csv(output / "forecast/forecast_metrics_extended.csv")
    later = metrics.loc[
        metrics["sample"].eq("retrospective_test") & metrics.promotion.eq(-1)
    ]
    selected = later.merge(
        chosen[["product", "model"]], on=["product", "model"]
    ).sort_values("product")
    robust = pd.read_csv(output / "pricing/robust_price_recommendations.csv")
    robust_aggregate = pd.read_csv(output / "pricing/robust_policy_aggregate.csv")
    covariance = pd.read_csv(output / "pricing/uncertainty_covariance_audit.csv")
    pricing = pd.read_csv(output / "pricing/pricing_sensitivity_aggregate.csv")
    scope = json.loads((output / "market_definition.json").read_text())[
        "optimized_scope"
    ]
    sample_coverage = pd.read_csv(output / "forecast/forecast_sample_coverage.csv")
    total = int(categories.raw_rows.sum())
    static = elasticities.loc[
        elasticities.system.eq("linked_product")
        & elasticities.specification.eq("week_FE")
    ]
    own_static = static.loc[
        static.price.eq((static.demand > 2).astype(int) + 1)
    ].sort_values("demand")
    own_dynamic = dynamics.loc[
        dynamics.system.eq("linked_product")
        & dynamics.price.eq((dynamics.demand > 2).astype(int) + 1)
    ]
    dynamic_table = own_dynamic.pivot(
        index="demand", columns="response", values="elasticity"
    )
    cross = elasticities.loc[
        elasticities.system.eq("product")
        & elasticities.specification.eq("week_FE")
        & elasticities.demand.ne(elasticities.price)
    ]
    category_rows = [
        [r.category, f"{int(r.raw_rows):,}", f"{int(r.valid_rows):,}", int(r.stores)]
        for r in categories.itertuples()
    ]
    coverage_rows = [
        [
            {1: "Training", 2: "Validation", 3: "Later evaluation"}[int(r.split)],
            f"{100*r.market_coverage:.1f}%",
            f"{100*r.full_category_coverage:.1f}%",
        ]
        for r in coverage.itertuples()
    ]
    dynamic_rows = [
        [
            NAMES[j - 1],
            f"{dynamic_table.loc[j,'immediate']:.2f}",
            f"{dynamic_table.loc[j,'subsequent']:.2f}",
            f"{dynamic_table.loc[j,'cumulative']:.2f}",
        ]
        for j in range(1, 5)
    ]
    forecast_rows = [
        [NAMES[int(r.product) - 1], r.model, f"{r.WAPE:.2f}%", f"{r.bias_pct:+.2f}%"]
        for r in selected.itertuples()
    ]
    scenario = pricing.loc[
        pricing.system.eq("linked_product")
        & pricing.policy.eq("joint")
        & pricing.maxchange.eq(0.15)
        & pricing.cost_multiplier.eq(1)
    ].copy()
    pricing_rows = [
        [r.specification, f"{r.gain_pct:.1f}%", f"{r.mean_price_change:.1f}%"]
        for r in scenario.itertuples()
    ]
    summary = (
        f"The complete run reads {total:,} scanner rows and selects four dominant 2L colas using training data. "
        f"They cover {100*coverage.iloc[0].market_coverage:.1f}% of broader 2L-cola training revenue, so optimization concerns a portfolio. "
        "Linked brand-price responses are better supported than separate regular/diet responses. "
        "Calibration reduces some forecast bias but does not uniformly improve later accuracy. "
        "Pricing gains remain conditional on observational demand and accounting-cost assumptions."
    )
    market = (
        f"The retained panel contains {audit['final_markets']:,} store-weeks in {audit['stores']} stores. "
        "Training is weeks 1-279; validation is 280-339; later evaluation is 340-399. "
        "Plain regular/diet 2L colas across brands enter training-revenue ranking, subject to 50% market observation coverage. "
        "The selected UPCs are 1200000230, 1200000231, 4900000639 and 4900000663. "
        "The broader market includes private labels, RC/Diet Rite cola, caffeine-free, cherry and clear colas. "
        "Fruit-flavored Diet Rite, cola champagne and generic multi-product Coke descriptions are excluded. "
        "Coverage denominators include all valid source stores, including stores outside the balanced panel."
    )
    construction = (
        "Unit price equals PRICE/QTY, revenue equals unit price times MOVE, and unit accounting cost equals "
        "unit price times (1-PROFIT/100). Hexadecimal fields preserve source double precision. "
        "OK=0, unusable keys, negative movement and nonpositive price/quantity are excluded. "
        "Missing markets and zero-price records are not converted into zero demand. "
        "Invalid accounting margins remove cost/profit calculations only. All valid raw UPCs match metadata; "
        "complete valid keys are unique. Cereal and detergent are fully audited and supply lagged store-activity signals "
        "in a separate matched-sample forecast sensitivity. Their package counts are not comparable physical quantities. "
        "Customer counts are purchasing visits; weekly totals require seven valid positive days. They do not track households."
    )
    identification = (
        f"After store and week effects and other prices/promotions, independent price standard deviations are "
        f"only {prices.independent_log_price_sd.min():.3f}-{prices.independent_log_price_sd.max():.3f} log points. "
        "The main demand system regresses each product's log sales on two brand price indices with frozen training revenue weights; "
        "the main policy changes regular/diet prices proportionally within brand. "
        "Unrestricted four-price and aggregate-family systems remain robustness checks. "
        "No signs or instruments are imposed. Inference allows arbitrary within-store and within-week dependence. "
        f"Of 12 unrestricted cross-price coefficients under week effects, {int(cross.upper95.lt(0).sum())} are significantly negative "
        f"and {int(cross.lower95.gt(0).sum())} significantly positive. This is not evidence establishing complementarity. "
        "Price endogeneity, coordinated promotions, incomplete promotion flags and omitted shocks remain."
    )
    experiments = (
        "The Kilts Center and Abrams, Gui and Hortacsu (2018) document pricing experiments, including soft drinks. "
        "Neither the attachments nor the documentation inspected provides verified treatment labels for these observations. "
        "A limited public search did not locate a verified downloadable assignment file. Price patterns, zones and sale codes "
        "are not treated as experimental assignments. A causal extension needs original store-category labels, dates, UPC coverage and implementation records."
    )
    dynamics_text = (
        f"A distributed-lag specification uses contemporaneous prices and four prior calendar weeks of prices and promotions "
        f"on {int(own_dynamic.N.iloc[0]):,} markets. Its immediate, subsequent and cumulative estimates are compared with a static model on that identical sample. "
        "The cumulative coefficients describe a persistent price change in this finite-lag model. Brand own-price responses remain negative, "
        "and lagged effects do not overturn their direction. Uncertainty is material; these associations do not distinguish stockpiling, habit formation or retailer response. "
        "No household histories or consumer inventory state are observed. Lead-price associations are exported as confounding/anticipation diagnostics."
    )
    pricing_text = (
        "The reference is week 399, with 76 stores jointly usable across demand specifications. Baseline quantities are model-fitted levels, "
        "so even the observed-price profit baseline is simulated. Joint accounting gross contribution is the sum of (price-cost) times predicted units, "
        "including every fitted cross-price response. Price movement caps are 5%, 10% and 15%; cost multipliers are 0.9, 1.0 and 1.1. "
        "Store-level 5th/95th price and all six relative-price percentiles constrain search; historical bounds explicitly expand to retain the observed baseline "
        "when it lies outside the central range. The linked grid is exhaustively searched. Unrestricted block-coordinate search is local; "
        "global finite-grid comparisons pass at the exported audit examples, without certifying continuous global optimality. "
        "Low accounting margins make raises attractive in these models: an own-contribution derivative has sign 1 + elasticity times margin, "
        "while the portfolio derivative adds cross-product terms. Upper-bound solutions are therefore sensitive to the permitted movement cap."
    )
    uncertainty = (
        f"{int(covariance.draws.iloc[0])} joint coefficient draws per model use the store/week clustered covariance, preserving cross-equation dependence "
        f"and leaving signs unrestricted. {int(covariance.clipped_eigenvalues.sum())} covariance eigenvalues required truncation to zero. "
        "Draws are shared across stores when aggregating gains. Baseline demand levels, market selection, historical support and costs remain conditional; "
        "this is not a confidence interval for realized business profit. The exploratory robust screen chooses a linked policy with nonnegative worst-model "
        f"5th-percentile draw gains across all {len(covariance)} estimated systems and all {len(config['pricing']['cost_multipliers'])} cost assumptions. "
        f"It passes in {int(robust.robust_screen_pass.sum())} of {len(robust)} stores. Mean changes are {robust.pepsi_change_pct.mean():+.2f}% for Pepsi "
        f"and {robust.coke_change_pct.mean():+.2f}% for Coke. Aggregate point gains span {robust_aggregate.point_gain_pct.min():.1f}% to "
        f"{robust_aggregate.point_gain_pct.max():.1f}% across these conditional scenarios. The same draws screen many candidates, so this is "
        "an exploratory model stress test, not a simultaneous inferential guarantee or an implementable causal recommendation."
    )
    accounting = (
        "PROFIT records accounting margins against average acquisition cost, which can differ from replacement or marginal cost. "
        "These simulations omit operating expenses, inventory financing, store traffic effects, retailer/manufacturer responses and profits on omitted substitutes. "
        "Their objective is portfolio accounting gross contribution, not whole-category economic profit. A small price increase that survives this screen "
        "is a candidate for a controlled experiment, not evidence of a free profit gain."
    )
    forecasts = (
        "Eight main candidates share the same eligible scoring rows. Calendar and recent-sales forecasts provide benchmarks; price-aware forecasts "
        "assume target-week posted prices and promotion plans are known at the origin. Rolling fits use only the preceding 156 weeks. "
        "Log(1+sales) predictions are retransformed using fitting-window smearing before subtracting one and clipping at zero. "
        "Pooled and promotion-specific calibration use only earlier one-step out-of-sample errors from the previous 26 weeks; factors shrink toward one "
        "and remain between 0.5 and 2.0. The minimum validation WAPE selects the final model, without switching based on later results."
    )
    calibration_text = (
        "Calibration often reduces underprediction but can increase WAPE. The selected Coke forecast still materially underpredicts later sales. "
        "The full tables retain these failures, promotion splits, weekly bias and training-defined price bins. "
        f"The supplemental cereal/detergent/traffic model is scored on {int(sample_coverage.iloc[0].activity_score_rows):,} rows "
        f"against the base model on exactly those rows; it is excluded from main selection because its information-set sample is smaller "
        f"than the {int(sample_coverage.iloc[0].common_score_rows):,} main scoring rows. "
        "Both validation and later periods had been inspected in earlier work, so all revised evaluation is retrospective. "
        "Chronological code prevents future-data leakage, but a new untouched period is still needed for prospective validation."
    )
    verification_text = (
        f"The complete raw run records {verification['checks']:,} verification checks: {verification['passed']:,} pass, "
        f"{verification['failed']} fail and {verification['warnings']} warnings. Checks cover raw totals and keys, training-only selection/weights, "
        "calendar lags, independent dummy-regression agreement for coefficients and two-way covariance, retransformation, model selection, "
        "future-outcome perturbations, price support, joint profit dominance, search examples and direct profit identities. "
        "Default numerical tolerances are absolute 1e-6 and relative 1e-8, with tighter absorption/linkage tolerances. "
        "Legacy unrestricted point estimates agree within tolerance; broader coverage, two-way inference, linked main pricing, purchasing dynamics "
        "and forecast calibration intentionally change other results. Windows was not executed. Experimental assignments and a fresh forecast test period are unavailable."
    )
    conclusion = (
        "The work supports a reproducible pricing-and-forecasting research workflow and documents substantial specification sensitivity. "
        "It supports studying linked brand prices and continuing calibration work. It does not establish causal substitution or achievable economic-profit gains."
    )
    sections = [
        ("Question and result", summary),
        ("Raw data and market", market),
        ("Construction", construction),
        ("Demand identification", identification),
        ("Experimental assignments", experiments),
        ("Purchasing dynamics", dynamics_text),
        ("Conditional pricing", pricing_text),
        ("Uncertainty and stress screen", uncertainty),
        ("Accounting interpretation", accounting),
        ("Forecasts", forecasts),
        ("Remaining calibration limits", calibration_text),
        ("Verification", verification_text),
        ("Economic conclusion", conclusion),
    ]
    title = "Retail Pricing with Product Substitution: Demand Estimation, Category Profit Optimization, and Sales Forecasting"
    tables = {
        "Raw data and market": markdown_table(
            ["Category", "Raw rows", "Valid rows", "Stores"], category_rows
        )
        + "\n\n"
        + markdown_table(
            ["Period", "Broader 2L cola coverage", "Full soft-drink coverage"],
            coverage_rows,
        ),
        "Purchasing dynamics": markdown_table(
            ["Product", "Immediate", "Subsequent", "Cumulative"], dynamic_rows
        ),
        "Conditional pricing": markdown_table(
            [
                "Linked specification",
                "Simulated gain, 15% cap / AAC",
                "Mean price change",
            ],
            pricing_rows,
        ),
        "Forecasts": markdown_table(
            ["Product", "Validation-selected model", "Later WAPE", "Later bias"],
            forecast_rows,
        ),
    }
    document = (
        "# "
        + title
        + "\n\n*Completed raw run; all revised evaluation is retrospective.*\n\n"
    )
    for header, paragraph in sections:
        document += "## " + header + "\n\n" + paragraph + "\n\n"
        if header in tables:
            document += tables[header] + "\n\n"
    document += (
        "## Sources and acknowledgement\n\nData provided by the James M. Kilts Center for Marketing at the University of Chicago Booth School of Business; "
        "academic research use. Dominick's manual, especially pages 3, 4, 7, 10-11 and 19. "
        "[Dataset documentation](https://www.chicagobooth.edu/research/kilts/research-data/dominicks). "
        "Abrams, Gui and Hortacsu (2018), [Finding Exogenous Variation in Data](https://arxiv.org/html/1704.07787v3), Section 4 and Appendix C. "
        "Source documentation accessed 2026-10-09. All reported numerical findings are generated from the supplied data.\n"
    )
    (reports / "ECONOMIC_REPORT.md").write_text(document)
    make_figures(output, reports)
    pdf_report(
        reports,
        title,
        sections,
        category_rows,
        coverage_rows,
        dynamic_rows,
        pricing_rows,
        forecast_rows,
        verification,
    )


def pdf_report(
    reports,
    title,
    sections,
    category_rows,
    coverage_rows,
    dynamic_rows,
    pricing_rows,
    forecast_rows,
    verification,
):
    # Embed Matplotlib's bundled fonts so native PDF font substitution cannot
    # change glyph spacing on another platform. No external font install is needed.
    font_directory = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(
        TTFont("ReportSans", str(font_directory / "DejaVuSans.ttf"))
    )
    pdfmetrics.registerFont(
        TTFont("ReportSansBold", str(font_directory / "DejaVuSans-Bold.ttf"))
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            fontName="ReportSansBold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor(BLUE),
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Deck",
            fontName="ReportSans",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor(GREEN),
            spaceAfter=15,
        )
    )
    styles["BodyText"].fontName = "ReportSans"
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 12.5
    styles["BodyText"].spaceAfter = 9
    styles["Heading2"].fontSize = 13
    styles["Heading2"].fontName = "ReportSansBold"
    styles["Heading2"].leading = 17
    styles["Heading2"].textColor = colors.HexColor(BLUE)
    styles["Heading2"].spaceBefore = 11
    sections = dict(sections)
    story = []

    def paragraph(value):
        return Paragraph(escape(value), styles["BodyText"])

    def section(name):
        story.extend(
            [Paragraph(escape(name), styles["Heading2"]), paragraph(sections[name])]
        )

    def table(headers, rows, widths):
        data = [[paragraph(x) for x in headers]] + [
            [paragraph(str(x)) for x in row] for row in rows
        ]
        result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7eff3")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(BLUE)),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f5f8fa")],
                    ),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )
        story.extend([result, Spacer(1, 8)])

    story.append(
        Paragraph("Retail pricing with<br/>product substitution", styles["ReportTitle"])
    )
    story.append(
        Paragraph(
            "Demand estimation, portfolio profit simulations and sales forecasting | October 2026",
            styles["Deck"],
        )
    )
    section("Question and result")
    section("Raw data and market")
    table(
        ["Category", "Raw rows", "Valid rows", "Stores"],
        category_rows,
        [75, 135, 135, 60],
    )
    table(
        ["Period", "2L cola coverage", "Full soft-drink coverage"],
        coverage_rows,
        [150, 130, 150],
    )
    section("Construction")
    story.append(PageBreak())
    section("Demand identification")
    section("Experimental assignments")
    section("Purchasing dynamics")
    table(
        ["Product", "Immediate", "Subsequent", "Cumulative"],
        dynamic_rows,
        [155, 90, 90, 95],
    )
    story.append(
        Image(
            str(reports / "figures/demand_dynamics.png"),
            width=6.3 * inch,
            height=2.75 * inch,
        )
    )
    story.append(PageBreak())
    section("Conditional pricing")
    story.append(
        Paragraph(
            "Point simulations: 15% movement cap, observed AAC costs", styles["Deck"]
        )
    )
    table(
        ["Linked specification", "Point gross-contribution gain", "Mean price change"],
        [[r[0].replace("_", " "), r[1], r[2]] for r in pricing_rows],
        [155, 145, 120],
    )
    section("Uncertainty and stress screen")
    section("Accounting interpretation")
    story.append(PageBreak())
    section("Forecasts")
    forecast_rows = [
        [
            r[0],
            r[1]
            .replace("rolling_promo_calibrated", "Promotion-calibrated rolling")
            .replace("rolling_no_trend", "Rolling, no trend"),
            r[2],
            r[3],
        ]
        for r in forecast_rows
    ]
    table(
        ["Product", "Selected model", "WAPE", "Bias"], forecast_rows, [120, 170, 65, 65]
    )
    section("Remaining calibration limits")
    story.append(
        Image(
            str(reports / "figures/forecast_calibration.png"),
            width=6.3 * inch,
            height=2.4 * inch,
        )
    )
    story.append(PageBreak())
    section("Verification")
    section("Economic conclusion")
    story.append(Paragraph("Sources and acknowledgement", styles["Heading2"]))
    story.append(
        paragraph(
            "Data: James M. Kilts Center for Marketing, University of Chicago Booth School of Business. Academic research use. "
            "Dominick's manual, pages 3-4, 7, 10-11 and 19; chicagobooth.edu/research/kilts/research-data/dominicks. "
            "Abrams, Gui and Hortacsu (2018), Finding Exogenous Variation in Data, arxiv.org/abs/1704.07787. Sources accessed 2026-10-09. "
            "Computed estimates and verification records are included in the repository."
        )
    )

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#d9e2e8"))
        canvas.line(45, 40, 567, 40)
        canvas.setFont("ReportSans", 8)
        canvas.setFillColor(colors.HexColor(BLUE))
        canvas.drawString(
            45, 27, "DOMINICK'S | Observational demand and conditional pricing"
        )
        canvas.drawRightString(567, 27, str(doc.page))

    doc = SimpleDocTemplate(
        str(reports / "Economic_Report.pdf"),
        pagesize=(612, 792),
        rightMargin=45,
        leftMargin=45,
        topMargin=43,
        bottomMargin=54,
        title=title,
        author="Retail Pricing Research Project",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
