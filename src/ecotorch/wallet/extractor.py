import json
from pathlib import Path


class Extractor:
    def __init__(self) -> None:
        base_path = Path(__file__).parent
        file_path = base_path / "electricity_price.json"
        self.prices = self._load_json(file_path)

    def _load_json(self, path) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError("Internal error, please contact the developers.")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                "Internal error, please contact developers.", e.doc, e.pos
            )
        return data

    def calculate_cost(self, country_code, kwh) -> float:
        return round(self._get_price(country_code) * kwh, 4)

    def _get_price(self, iso3) -> float:
        return self.prices[iso3]
