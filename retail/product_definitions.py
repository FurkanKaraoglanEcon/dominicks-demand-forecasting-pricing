"""Visible four-slot schema; the selected UPC mapping is exported every run.

For the cola portfolio slots are Pepsi, Diet Pepsi, Coke and Diet Coke.
For cereal, slots follow training revenue rank. Detergent slots follow the
declared UPC portfolio in config.toml. They represent package formats rather
than assumed homogeneous physical goods.

Inputs: no data; explicit schema and cola definitions used by stage modules.
Outputs: stable slot, column-name, UPC and label constants.
"""

PRODUCTS = ["p1", "p2", "p3", "p4"]
PRICE_NAMES = ["price_p1", "price_p2", "price_p3", "price_p4"]
PROMOTION_NAMES = ["promotion_p1", "promotion_p2", "promotion_p3", "promotion_p4"]
COLA_UPCS = {"p1": 1200000230, "p2": 1200000231, "p3": 4900000639, "p4": 4900000663}
COLA_LABELS = {
    "p1": "Pepsi",
    "p2": "Diet Pepsi",
    "p3": "Coca-Cola Classic",
    "p4": "Diet Coke",
}
