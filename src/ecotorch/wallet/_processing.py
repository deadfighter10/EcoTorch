from pathlib import Path
import pandas as pd
from pprint import pprint
import json

import pycountry

data = {}
price_units = {}
conversion = {
    "EUR": 1.17955,
    "INR": 0.010992,
    "AUD": 0.705465,
    "GBP": 1.348895,
    "BGN": 0.60,
    "WON": 0.00069,
    "JPY": 0.0065,
    "CAD": 0.73,
    "BRL": 0.19,
    "SGD": 0.79,
    "RON": 0.23,
    "USD": 1
}

def convert_country_to_iso(country_name: str):
    try:
        result = pycountry.countries.search_fuzzy(country_name)[0]
        return result.alpha_3
    except (LookupError, IndexError):
        return "World"

info_table = pd.read_csv("./src/ecotorch/wallet/infotable.csv")
p = Path("./src/ecotorch/wallet/datasets")
for file_path in p.glob('*.csv'):
    df = pd.read_csv(file_path)
    name = str(file_path.name).split("_")[0]
    mask = (info_table['Country'] == name)
    try:
        unit = str(info_table.loc[mask, "Price Unit"].iloc[0])
        price_units[name] = unit.split("/")
    except IndexError:
        print(name)

    try:
        value = float(df[name].tail(1).item())
    except KeyError:
        value = float(df.iloc[-1, 1])

    if price_units[name][1] == "MWh":
        value /= 1000

    value *= conversion[price_units[name][0]]

    name = convert_country_to_iso(name)

    data[name] = value


data["World"] = 0.162
base_path = Path(__file__).parent
file_path = base_path / "electricity_price.json"

# Open the file in write mode and dump the data
with open(file_path, 'w') as json_file:
    json.dump(data, json_file, indent=4)
