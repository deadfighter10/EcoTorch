from pathlib import Path
import pandas as pd

class DataHandler:
    def __init__(self):
        base_path = Path(__file__).parent
        csv_path = base_path / 'co2emission.csv'
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['year'] == 2024]

    def get_value_by_country(self, country) -> float:
        if len(self.df[self.df['country'] == country]['co2_per_unit_energy']) == 0:
            raise ValueError("Invalid country name!")

        return float(self.df[self.df['country'] == country]['co2_per_unit_energy'].item())

    def get_countries_list(self) -> list:
        return self.df['country'].tolist()

