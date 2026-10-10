"""Write a concise economic report from current-run tables and verification.

Inputs: generated coverage, demand, pricing and forecast results. Outputs:
Markdown/PDF report, research figures and a component-completion checklist.
All numerical statements are computed here, rather than frozen in prose.
"""

from xml.sax.saxutils import escape
from pathlib import Path
import json
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from .report_figures import make_figures
from .io import save_csv

TITLE = "Retail Pricing with Product Substitution: Demand Estimation, Category Profit Optimization, and Sales Forecasting"


def markdown_table(headers, rows):
    """Render a small evidence table without altering stored numerical precision."""
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def report_sections(config, verification):
    """Assemble findings and limitations using this run's results and settings."""
    output = config["paths"]["output"]
    audit = json.loads((output / "preparation_audit.json").read_text())
    categories = pd.read_csv(output / "categories/category_summary.csv")
    coverage = pd.read_csv(output / "revenue_coverage.csv")
    pricing_notes = json.loads((output / "pricing/pricing_notes.json").read_text())
    prices = pd.read_csv(output / "identification/independent_price_variation.csv")
    dynamics = pd.read_csv(output / "demand/demand_dynamics.csv")
    primary = pd.read_csv(output / "pricing/pricing_sensitivity_aggregate.csv")
    main = primary[
        (primary.system == "linked_product")
        & (primary.specification == "week_FE")
        & (primary.policy == "joint")
        & (primary.maxchange == max(config["pricing"]["price_changes"]))
        & (primary.cost_multiplier == 1)
    ].iloc[0]
    robust = pd.read_csv(output / "pricing/robust_price_recommendations.csv")
    forms = pd.read_csv(output / "pricing/common_policy_functional_forms.csv")
    forecasts = pd.read_csv(output / "tables/forecast_test_metrics.csv")
    winners = forecasts[forecasts.selected_before_test].sort_values(
        ["category", "information_set"]
    )
    windows = pd.read_csv(output / "tables/forecast_validation_windows.csv")
    unavailable = pd.read_csv(output / "tables/forecast_information_sets.csv")
    unavailable = unavailable[
        unavailable.information_set == "selected_products_not_estimable"
    ]
    settings = config["sample"]
    sections = []
    sections.append(
        (
            "Research question and findings",
            f"How do joint price changes affect sales and accounting gross contribution when closely related products share pricing decisions? "
            f"The pipeline reads {int(categories.raw_rows.sum()):,} scanner rows. Four plain regular/diet 2L colas account for {100*coverage.iloc[0].market_coverage:.1f}% of broader 2L-cola training revenue. "
            f"The optimized market is therefore a four-product portfolio. The linked-price main model implies a {main.gain_pct:.1f}% gross-contribution gain at a {100*main.maxchange:.0f}% movement cap and recorded acquisition costs. "
            "That number is a conditional simulation, not an identified effect of a retail intervention. Forecast comparisons preserve chronological information sets, but all evaluation periods have been inspected before and remain retrospective.",
        )
    )
    rows = [
        [r.category, f"{int(r.raw_rows):,}", f"{int(r.valid_rows):,}"]
        for r in categories.itertuples()
    ]
    sections.append(
        (
            "Data and market",
            f'The primary panel has {audit["final_markets"]:,} store-weeks and {audit["stores"]} stores. Training-revenue ranking chooses Pepsi, Diet Pepsi, Coca-Cola Classic and Diet Coke; at least {config["data"]["minimum_training_weeks"]} complete training weeks per store are required. '
            "The broader substitute set includes other brands, private labels, cherry, caffeine-free and clear colas. Ambiguous descriptions are disclosed. Cereal is selected by training revenue, while detergent retains the uploaded report's declared UPC portfolio with training eligibility audited; their differing package sizes prevent a common physical-volume interpretation. Whole-file package totals support additional forecasting targets.\n\n"
            + markdown_table(["File category", "Raw rows", "Valid priced rows"], rows),
        )
    )
    sections.append(
        (
            "Measurement and identification",
            "Package price is PRICE/QTY; revenue is that price times MOVE. Recorded PROFIT implies acquisition cost p(1-PROFIT/100). Negative gross margins remain valid when implied cost is positive and finite; unusable costs do not remove demand rows. Missing records and unpriced zero-sales rows are never treated as observed zero demand. "
            "Weekly customer counts require seven valid nonnegative daily observations and a positive weekly total. Counts measure purchasing visits, not tracked households. Census income is already logged. "
            f"After store/week effects, rival prices and recorded promotions, independent price standard deviations range from {prices.independent_log_price_sd.min():.3f} to {prices.independent_log_price_sd.max():.3f}. "
            "The main system uses two frozen training-revenue brand price indices; regular/diet decisions change proportionally within brand. Unrestricted four-price and aggregate-family systems remain robustness checks. No substitution signs are forced. "
            "Two-way store/week covariance preserves serial dependence and common week shocks. Documented experiments do not supply verified treatment assignments in these inputs. AAC-based 2SLS is an explicitly unverified exclusion sensitivity; other-store prices are diagnosed, not presumed valid instruments.",
        )
    )
    own = dynamics[
        (dynamics.system == "linked_product")
        & (dynamics.price == ((dynamics.demand > 2).astype(int) + 1))
    ].pivot(index="demand", columns="response", values="elasticity")
    subsequent = dynamics[
        (dynamics.system == "linked_product")
        & (dynamics.price == ((dynamics.demand > 2).astype(int) + 1))
        & (dynamics.response == "subsequent")
    ]
    intervals_include_zero = (
        (subsequent.lower95 <= 0) & (subsequent.upper95 >= 0)
    ).sum()
    rows = [
        [
            label,
            f'{own.loc[j,"immediate"]:.2f}',
            f'{own.loc[j,"subsequent"]:.2f}',
            f'{own.loc[j,"cumulative"]:.2f}',
        ]
        for j, label in enumerate(["Pepsi", "Diet Pepsi", "Coke", "Diet Coke"], 1)
    ]
    sections.append(
        (
            "Demand dynamics and scope",
            "Static models with week effects and alternative seasonal/trend controls are compared with current plus four calendar lags of prices and promotions on matched samples. Immediate, subsequent and cumulative responses distinguish timing sensitivity; cumulative slopes describe a persistent price change in the finite-lag model. "
            "Separate lagged-quantity models report persistence and delta-method long-run effects, with fixed-effects endogeneity acknowledged. Other executed checks cover omitted-cola indices, log1p zeros, semilog prices, income interactions, traffic controls, per-visit outcomes, brand and diet family partitions, category-volume responses, lead prices and descriptive event windows. These associations do not distinguish inventory accumulation from habit or retailer responses.\n\n"
            f"The linked own-brand subsequent responses range from {subsequent.elasticity.min():.2f} to {subsequent.elasticity.max():.2f}; {intervals_include_zero} of four 95% intervals include zero. The point estimates show limited additional response within four weeks, while cumulative and static responses remain separate pricing stress specifications.\n\n"
            + markdown_table(
                ["Product", "Immediate", "Subsequent", "Cumulative"], rows
            ),
        )
    )
    sections.append(
        (
            "Pricing interpretation and numerical search",
            f'The main reference is week {pricing_notes["reference_week"]} with {int(main.stores)} jointly usable stores. Baseline quantities are fitted means, so observed-price baseline profit is itself simulated. '
            "The objective sums (price-cost) times all four predicted quantities, including every cross-price response. It measures portfolio accounting gross contribution; it excludes operating expenses, inventory finance, manufacturer responses, store traffic effects and omitted-substitute profits. It is not economic profit. "
            "Movement caps, marginal-price support and all six historical log-price-ratio ranges constrain policies. Explicit exceptions retain a reference outside the central historical range. The independent benchmark holds rivals at baseline, then shrinks the assembled log changes if needed for a feasible simultaneous rollout. "
            "Linked decisions have an exhaustive finite-grid search. Unrestricted block search is local and audited against finite-grid examples; representative functional-form checks also enumerate a four-price grid and attempt local continuous refinement. Neither establishes a global continuous optimum. Cereal/detergent simulations cover package portfolios. Representative-market gains are not directly comparable with aggregate store-reference gains.",
        )
    )
    sections.append(
        (
            "Uncertainty and common-policy robustness",
            f'The primary screen uses {config["pricing"]["draws"]} joint two-way clustered Gaussian slope draws, shared across stores, with costs multiplied by {", ".join(map(str,config["pricing"]["cost_multipliers"]))}. '
            f"A positive worst-model lower-tail screen passes for {int(robust.robust_screen_pass.sum())} of {len(robust)} stores. The same policy is then evaluated under log-log, log1p and semilog objectives: the smallest additional lower-tail aggregate gain is {forms.gain_p05.min():.2f} in scanner dollars. "
            f"Screened store-average linked price changes are {robust.pepsi_change_pct.mean():.2f}% for Pepsi/Diet Pepsi and {robust.coke_change_pct.mean():.2f}% for Coke/Diet Coke. "
            "Negative covariance eigenvalues, if present, are disclosed before a positive-semidefinite drawing adjustment. Baseline levels, selected products, historical support and acquisition costs are held fixed. These are exploratory model stress tests, not simultaneous confidence guarantees or evidence that a price change will yield the simulated gain. A robust experiment candidate must also pass the additional functional-form checks; the output labels these separately.",
        )
    )
    category_labels = {"cer": "Cereal", "lnd": "Detergent", "sdr": "Soft drinks"}
    information_labels = {
        "operational": "Operational",
        "conditional_current_policy": "Current-policy conditional",
        "operational_full_category": "Full category",
    }
    model_labels = {
        "boosting_15": "Boosting (15)",
        "boosting_31": "Boosting (31)",
        "forest_16": "Forest (16)",
        "forest_32": "Forest (32)",
        "ridge_1": "Ridge (1)",
        "ridge_10": "Ridge (10)",
        "ridge_100": "Ridge (100)",
        "ridge_calibrated": "Calibrated ridge",
        "seasonal_52_weeks": "Annual seasonal",
        "mean_four_weeks": "Four-week mean",
        "last_week": "Last week",
    }
    rows = [
        [
            category_labels[r.category],
            information_labels[r.information_set],
            model_labels.get(r.model_id, r.model_id),
            f"{100*r.WAPE:.1f}%",
            f"{r.aggregate_ratio:.2f}",
            f"{100*r.interval_coverage:.1f}%",
        ]
        for r in winners.itertuples()
    ]
    sections.append(
        (
            "Operational forecasting and calibration",
            f'Operational training ends at week {settings["train_end"]}; candidate selection uses {settings["train_end"]+1}-{settings["selection_end"]}, interval calibration {settings["selection_end"]+1}-{settings["calibration_end"]}, and test evaluation {settings["calibration_end"]+1}-{settings["last_week"]}. '
            "Persistence, moving averages and annual seasonal benchmarks compete with ridge, random forest, histogram boosting and a chronologically calibrated ridge. Operational features use sales, traffic, prices and promotions through t-1; a separate conditional exercise adds realized current policy and is retrospective. Models refit at scheduled origins. Within a refit block, each forecast is still one week ahead, using sales observed by that week. "
            "Imputation, scaling and product smearing are fitted on past rows. Log1p means equal exp(prediction) times the training smearing factor minus one. Point calibration uses a held-out tail inside each training window; interval radii use the separate pretest period. Selection remains fixed before test results.\n\n"
            "Accuracy gains vary by target and information set. The selected cola operational model underpredicts aggregate sales; calibration does not uniformly improve test accuracy, and nominal 90% intervals do not attain 90% coverage in every exercise. These discrepancies remain in the results rather than prompting test-based model changes.\n\n"
            + markdown_table(
                [
                    "Category",
                    "Information set",
                    "Selected model",
                    "WAPE",
                    "Forecast / actual",
                    "Interval coverage",
                ],
                rows,
            ),
        )
    )
    sections.append(
        (
            "Forecast limitations and verification",
            f"Data gaps leave {int((windows.validation_rows==0).sum())} validation windows without eligible rows and prevent selected-product forecasting for {len(unavailable)} {'category' if len(unavailable)==1 else 'categories'}. "
            "Cereal has a source gap; detergent lacks a sustained complete four-UPC sample late in the study, although some individual UPC observations continue. Full-category forecasts remain separate targets. Seasonal lags can be missing and use documented fallback or past-only imputation. Bias is reported by product, promotion and training-defined price bins. "
            "Interval coverage is empirical under dependent retail data, not a distribution-free guarantee. The selected-product forecast population also requires realized target-week price availability; prospective assortment forecasting needs a planned stocking panel. The uploaded hyperparameter configuration was unavailable, so new defaults are disclosed rather than claimed as exact replication. "
            f'Verification records {verification["passed"]} passing checks, {verification["failed"]} failures and {verification["warnings"]} warnings. Meaningful tests cover independent dummy OLS and covariance, IV algebra, calendar gaps, retransformation, future perturbations, price feasibility, search and accounting identities. Tolerances and discrepancies are saved. No fresh untouched forecast period or verified experimental assignment is available.',
        )
    )
    sections.append(
        (
            "Data sources",
            "Dominick's scanner movement, UPC metadata, demographics, customer counts and supplied manual are provided by the James M. Kilts Center, University of Chicago Booth School of Business. The assignment audit cites Abrams, Gui and Hortacsu, Finding Exogenous Variation in Data (2018), while making no claim that those treatment labels are available here. Raw checksums record the exact supplied files.",
        )
    )
    return sections, winners


def write_pdf(path, sections):
    """Render the same report narrative and evidence tables for convenient reading."""
    fonts = Path(__file__).resolve().parent / "assets"
    pdfmetrics.registerFont(TTFont("ResearchSans", str(fonts / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(
        TTFont("ResearchSans-Bold", str(fonts / "DejaVuSans-Bold.ttf"))
    )
    styles = getSampleStyleSheet()
    styles["BodyText"].fontName = "ResearchSans"
    styles["Title"].fontName = "ResearchSans-Bold"
    styles["Heading2"].fontName = "ResearchSans-Bold"
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 12
    story = [Paragraph(escape(TITLE), styles["Title"]), Spacer(1, 12)]
    for heading, text in sections:
        story.append(Paragraph(escape(heading), styles["Heading2"]))
        for paragraph in text.split("\n\n"):
            if paragraph.startswith("|"):
                lines = paragraph.splitlines()
                rows = [
                    [cell.strip() for cell in line.strip("|").split("|")]
                    for line in [lines[0]] + lines[2:]
                ]
                cells = [
                    [Paragraph(escape(cell), styles["BodyText"]) for cell in row]
                    for row in rows
                ]
                table = Table(
                    cells, colWidths=[480 / len(rows[0])] * len(rows[0]), repeatRows=1
                )
                table.setStyle(
                    TableStyle(
                        [
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7eff3")),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                        ]
                    )
                )
                story.append(table)
            else:
                story.append(Paragraph(escape(paragraph), styles["BodyText"]))
            story.append(Spacer(1, 8))

    def footer(canvas, doc):
        canvas.setFont("ResearchSans", 8)
        canvas.drawRightString(550, 28, f"{doc.page}")

    SimpleDocTemplate(
        str(path),
        pagesize=(612, 792),
        leftMargin=54,
        rightMargin=54,
        topMargin=45,
        bottomMargin=45,
    ).build(story, onFirstPage=footer, onLaterPages=footer)


def write_checklist(config):
    """Tie completion claims to produced artifacts, including non-estimable windows."""
    output = config["paths"]["output"]
    components = [
        (
            "Complete raw scanner/metadata/demographic/customer preparation",
            "raw_checksums.json",
        ),
        (
            "Training-only cola selection and substitute coverage",
            "revenue_coverage.csv",
        ),
        ("Category portfolios and visible UPC mappings", "tables/product_mapping.csv"),
        (
            "Price variation, experimental audit and event profiles",
            "identification/event_profiles.csv",
        ),
        (
            "13 primary static/linked/family/omitted/dynamic systems",
            "demand/demand_sensitivity.csv",
        ),
        (
            "Functional forms, zeros, income, visits, persistence, AAC IV and category demand",
            "tables/demand_coefficients.csv",
        ),
        (
            "Linked/free/store cost and uncertainty scenarios",
            "pricing/pricing_sensitivity_aggregate.csv",
        ),
        (
            "Common policy stress under functional forms",
            "pricing/common_policy_functional_forms.csv",
        ),
        (
            "Store-only wild-score fixed-policy and reoptimized sensitivity",
            "pricing/store_wild_summary.csv",
        ),
        (
            "Representative category/form/cost/constraint scenarios",
            "pricing/functional_form_scenarios.csv",
        ),
        (
            "Retained research forecast candidates and activity sensitivity",
            "forecast/forecast_activity_sensitivity.csv",
        ),
        (
            "Operational/conditional/full-category forecasting and point/interval calibration",
            "tables/forecast_test_metrics.csv",
        ),
        ("Independent verification and tests", "verification_checks.csv"),
    ]
    rows = [
        dict(
            component=name,
            status="executed" if (output / path).exists() else "missing",
            evidence=path,
        )
        for name, path in components
    ]
    save_csv(rows, config["paths"]["reports"], "component_completion.csv")
    audit = pd.read_csv(output / "tables/forecast_information_sets.csv")
    unavailable = audit[audit.information_set == "selected_products_not_estimable"]
    text = "# Component completion\n\n" + markdown_table(
        ["Component", "Status", "Evidence"],
        [[r["component"], r["status"], r["evidence"]] for r in rows],
    )
    text += (
        "\n\nUnexecuted because not estimable from the supplied dated-lag sample:\n\n"
    )
    text += (
        "\n".join(
            f"- {r.category}: selected-product operational and conditional forecast; full-category forecast executed."
            for r in unavailable.itertuples()
        )
        or "None."
    )
    (config["paths"]["reports"] / "COMPONENT_COMPLETION.md").write_text(text + "\n")


def write_report(config, verification):
    """Export current-run economic narrative, figures and completion evidence."""
    sections, winners = report_sections(config, verification)
    reports = config["paths"]["reports"]
    reports.mkdir(parents=True, exist_ok=True)
    text = (
        "# "
        + TITLE
        + "\n\n"
        + "\n\n".join(
            "## " + heading + "\n\n" + paragraph for heading, paragraph in sections
        )
        + "\n"
    )
    (reports / "ECONOMIC_REPORT.md").write_text(text)
    make_figures(config["paths"]["output"], reports)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(
        winners.category + " / " + winners.information_set,
        100 * winners.WAPE,
        color="#173b58",
    )
    ax.set_xlabel("Retrospective test WAPE (%)")
    fig.tight_layout()
    fig.savefig(reports / "figures/operational_forecast_accuracy.png", dpi=180)
    plt.close(fig)
    write_pdf(reports / "ECONOMIC_REPORT.pdf", sections)
    write_checklist(config)
