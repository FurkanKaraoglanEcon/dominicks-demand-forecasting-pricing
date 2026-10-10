"""Full-panel own and portfolio gross-contribution derivatives.

Inputs: each unrestricted model's fitted mean sales and observed prices/AACs.
Outputs: product-level derivative records and sign/margin summaries. Cross-price
responses enter joint contribution; these are conditional diagnostic slopes.
"""

import numpy as np
import pandas as pd
from .io import save_csv
from .pricing import profit_gradient


def full_panel_derivatives(panel, models, output):
    """Evaluate own and joint price derivatives using panel costs and fitted models.
    Write product details and summaries to output; return no fitted model."""
    prices = panel[[f"p{j}" for j in range(1, 5)]].to_numpy(float)
    costs = panel[[f"cost{j}" for j in range(1, 5)]].to_numpy(float)
    tables = []
    for key, model in models.items():
        if model.system != "product":
            continue
        usable = np.isfinite(costs).all(axis=1) & np.isfinite(model.baseline_units).all(
            axis=1
        )
        p, c, q = prices[usable], costs[usable], model.baseline_units[usable]
        margin = (p - c) / p
        own = np.diag(model.elasticity)
        normalized = 1 + margin * own
        joint = profit_gradient(p, q, c, model.elasticity)
        implied = np.divide(-1, own, out=np.full(4, np.nan), where=own < 0)
        count = int(usable.sum())
        tables.append(
            pd.DataFrame(
                dict(
                    store=np.repeat(panel.loc[usable, "store"].to_numpy(), 4),
                    week=np.repeat(panel.loc[usable, "week"].to_numpy(), 4),
                    product=np.tile(np.arange(1, 5), count),
                    specification=key[1],
                    elasticity=np.tile(own, count),
                    margin=margin.reshape(-1),
                    implied_margin=np.tile(implied, count),
                    own_slope_normalized=normalized.reshape(-1),
                    own_profit_derivative=(q * normalized).reshape(-1),
                    portfolio_profit_derivative=joint.reshape(-1),
                    response="cumulative" if model.dynamic else "static",
                )
            )
        )
    details = pd.concat(tables, ignore_index=True)
    save_csv(details, output, "profit_derivative_details.csv")
    summary = (
        details.assign(
            own_positive=details.own_slope_normalized.gt(0),
            joint_positive=details.portfolio_profit_derivative.gt(0),
        )
        .groupby(["specification", "product"], as_index=False)
        .agg(
            N=("store", "size"),
            margin_mean=("margin", "mean"),
            implied_margin=("implied_margin", "first"),
            share_own_positive=("own_positive", "mean"),
            share_portfolio_positive=("joint_positive", "mean"),
        )
    )
    save_csv(summary, output, "profit_derivative_summary.csv")
