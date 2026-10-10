# Retail Pricing with Product Substitution

## Research question

How does coordinated pricing of related products affect sales and retail gross profit, and how accurately can those sales be forecast using information available at the forecast date?

This Python project combines scanner-data preparation, demand estimation, pricing simulations, and chronological forecast evaluation in one pipeline. The main pricing exercise covers four regular and diet 2L cola products. Cereal and laundry detergent provide additional demand and forecasting exercises. Product mappings and revenue coverage make the scope of each portfolio explicit.

## Analysis

The pipeline reads the complete movement archives and combines them with UPC metadata, store demographics, and customer counts. It constructs store-week panels, selects eligible cola and cereal products using training data, and audits a configured detergent portfolio. Customer counts measure aggregate store traffic.

Demand models estimate own-price and cross-price responses under alternative controls for store differences and common time effects. The analysis examines independent price variation, regular/diet price linkage, omitted substitutes, and immediate and subsequent sales responses. Inference accounts for dependence across stores and weeks. Additional specifications assess sensitivity to functional form, sample restrictions, and alternative identifying assumptions.

Pricing simulations use linked regular/diet changes as the main exercise and unrestricted product prices as a robustness check. Historical price support, relative-price constraints, alternative cost assumptions, and estimation uncertainty limit the candidate policies. Demand prediction, gross-profit calculation, and numerical price search are separate calculations.

Forecasting compares seasonal and recent-sales benchmarks with regression and tree models. Operational forecasts use information available at each forecast origin; forecasts conditional on current prices are reported separately. Chronological periods separate model selection, calibration, and final evaluation. The pipeline evaluates bias, retransformation, and prediction-interval coverage.

## Interpretation

Estimated price responses describe observational relationships; independent variation is limited for some closely linked products. Pricing gains are conditional simulations rather than demonstrated causal effects of a pricing intervention. When selected products do not adequately cover the category, results refer to the selected portfolio.

The profit measure is accounting gross profit based on sales revenue less measured acquisition cost. It does not include all operating costs or measure economic profit. Forecast evaluations use historical data and are retrospective. Selected-product detergent forecasts are reported as not estimable where the required common evaluation sample is unavailable; category-level forecasts remain a separate exercise.

## Repository structure

| File or folder | Purpose |
| --- | --- |
| `run.py` | Short entry point for the complete pipeline |
| `config.toml` | User-editable paths, sample periods, model settings, pricing scenarios, and numerical tolerances |
| `requirements.txt` | Python dependencies |
| `retail/` | Analysis modules organized by research stage |
| `tests/` | Tests of economically relevant calculations and information restrictions |
| `data/raw/` | Local raw inputs and an input inventory |
| `output/` | Generated datasets, estimates, simulations, forecast results, and verification records |
| `reports/` | Economic report, figures, and component-completion documentation |
| `METHODS.md` | Model specifications, sample definitions, and assumptions |
| `DATA_TRACKING.md` | Input checksums and guidance on tracking generated files |
| `.gitignore` | Rules excluding specified local data, intermediate files, environments, and caches from Git |

Within `retail/`, the stages are organized as follows:

| Stage | Main modules |
| --- | --- |
| Preparation and product definitions | `preparation.py`, `selection.py`, `category_panels.py`, `product_definitions.py` |
| Identification | `identification.py` |
| Demand estimation and inference | `demand.py`, `demand_extensions.py`, `econometrics.py`, `equation_inference.py` |
| Pricing and uncertainty | `pricing.py`, `pricing_scenarios.py`, `pricing_uncertainty.py`, `pricing_forms.py`, `pricing_bootstrap.py`, `profit_diagnostics.py` |
| Forecasting | `forecasting.py`, `forecast_benchmarks.py`, `forecast_operational.py` |
| Verification and reporting | `verification.py`, `verification_research.py`, `verification_extensions.py`, `reporting.py`, `report_figures.py` |

`pipeline.py` defines the stage order, and `io.py` provides file, calendar, and logging utilities. Module docstrings describe inputs and outputs. `retail/assets/` contains the fonts and license used to generate the PDF report.

## Run the project

Use Python 3.11 or newer. Run the following commands from the repository root:

```bash
python -m venv .venv
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies, run the tests, and execute the analysis:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python run.py --config config.toml
```

Place the three movement ZIP archives, three UPC metadata CSV files, `demo.dta`, `ccount.dta`, and `codebook.pdf` in `data/raw/`, using the filenames listed in `data/raw/README.md`. Keep the movement archives zipped. To use another input directory, edit `paths.raw` in `config.toml`. Configured paths resolve relative to the configuration file.

The complete run starts from raw inputs, executes every estimable component, and regenerates the results. It overwrites generated outputs while preserving raw files. Local data and intermediate datasets may be absent from the Git repository; they are recreated when the required inputs are available.

## Results and verification

Start with `reports/ECONOMIC_REPORT.pdf` for the economic interpretation and `reports/COMPONENT_COMPLETION.md` for component status. The principal numerical outputs are:

- Product mappings, selection audits, and revenue coverage in `output/` and `output/tables/`.
- Identifying-variation diagnostics in `output/identification/`.
- Demand estimates and dynamic responses in `output/demand/`.
- Pricing scenarios, robust-policy summaries, and search diagnostics in `output/pricing/`.
- Forecast benchmarks in `output/forecast/` and operational evaluation tables in `output/tables/`.
- Input checksums, execution settings, and verification checks in `output/raw_checksums.json`, `output/run_manifest.json`, `output/verification_report.json`, and `output/verification_checks.csv`.

Verification covers sample construction, training-only selection, estimation and inference, forecast information sets, model selection, retransformation, pricing feasibility, profit calculations, and numerical search behavior. Assertions use explicit tolerances defined in the configuration. The run manifest and component records distinguish completed calculations from exercises that cannot be estimated on the available sample.

## Version control

Track the complete source code, tests, configuration, dependencies, documentation, and selected aggregate results. Keep raw data, environments, model binaries, detailed predictions, and large intermediate datasets outside version control.

The root `.gitignore` applies automatically, but it is not a complete publication whitelist. Review the files being staged before committing. Files already tracked by Git remain tracked even after a matching ignore rule is added. Record methodological changes together with their regenerated results so that the code, settings, and reported findings remain consistent.
