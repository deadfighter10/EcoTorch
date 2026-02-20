from .geolocator import get_location, convert_country_to_iso

from pathlib import Path
import pandas as pd

class DataHandler:
    def __init__(self):
        base_path = Path(__file__).parent
        csv_path = base_path / 'co2intensity.csv'
        self.df = pd.read_csv(csv_path)
        self._country = get_location()

    def get_intensity(self, country: str = None) -> float:
        if not self._country or country:
            if country is None:
                _country_code = get_location()
            else:
                _country_code = convert_country_to_iso(country)
            if _country_code is None:
                raise ValueError("Cannot find country code!")
        else:
            _country_code = self._country

        if _country_code == "World":
            _latest_year = self._get_latest_year_for_world
            mask = (self.df['Area'] == _country_code) & (self.df['Variable'] == "CO2 intensity") & (self.df['Year'] == _latest_year)
            data = self.df.loc[mask, "Value"]

            if data.empty:
                raise ValueError("Couldn't find values for the world, please provide a country!")
            return float(data.iloc[0])

        _latest_year = self._get_latest_year_by_country_code(_country_code)
        if _latest_year is None:
            raise ValueError("No data for the country!")


        mask = (self.df['ISO 3 code'] == _country_code) & (self.df['Variable'] == "CO2 intensity") & (self.df['Year'] == _latest_year)
        data = self.df.loc[mask, "Value"]

        if data.empty:
            raise ValueError("Invalid country name!")
        return float(data.iloc[0])

    def _get_latest_year_by_country_code(self, country_code: str) -> int | None:
        _max_year = int(max(self.df["Year"].unique().tolist()))
        for i in range(_max_year, _max_year-4, -1):
            _mask = (self.df['ISO 3 code'] == country_code) & (self.df['Variable'] == "CO2 intensity") & (self.df['Year'] == i)
            _data = self.df.loc[_mask, "Value"]
            if _data.empty:
                continue
            return i
        return None

    @property
    def _get_latest_year_for_world(self) -> int | None:
        _max_year = int(max(self.df["Year"].unique().tolist()))
        for i in range(_max_year, _max_year-4, -1):
            _mask = (self.df['Area'] == "World") & (self.df['Variable'] == "CO2 intensity") & (self.df['Year'] == i)
            _data = self.df.loc[_mask, "Value"]
            if _data.empty:
                continue
            return i
        return None

    @property
    def countries(self) -> list:
        return sorted(self.df['Area'].unique().tolist())

