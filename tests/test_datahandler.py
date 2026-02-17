import pytest
from unittest.mock import patch, MagicMock
from ecotorch.datahandler import DataHandler
import pandas as pd

@pytest.fixture
def mock_df():
    data = {
        'Area': ['United States', 'United States', 'Austria'],
        'ISO 3 code': ['USA', 'USA', 'AUT'],
        'Variable': ['CO2 intensity', 'CO2 intensity', 'CO2 intensity'],
        'Year': [2020, 2021, 2021],
        'Value': [0.4, 0.35, 0.2]
    }
    return pd.DataFrame(data)

@pytest.fixture
def data_handler(mock_df):
    with patch('pandas.read_csv', return_value=mock_df):
        handler = DataHandler()
        return handler

def test_get_intensity_mocked_location(data_handler):
    with patch('ecotorch.datahandler.get_location', return_value='USA'):
        assert data_handler.get_intensity() == 0.35

def test_get_intensity_specific_country(data_handler):
    with patch('ecotorch.datahandler.convert_country_to_iso', return_value='USA'):
        assert data_handler.get_intensity('United States') == 0.35

def test_get_intensity_invalid_country_code(data_handler):
    with patch('ecotorch.datahandler.convert_country_to_iso', return_value=None):
        with pytest.raises(ValueError, match="Cannot find country code"):
            data_handler.get_intensity('NonExistent')

def test_get_intensity_no_data(data_handler):
    with patch('ecotorch.datahandler.convert_country_to_iso', return_value='ZZZ'):
        # ZZZ is not in mock_df
        with pytest.raises(ValueError, match="No data for the country"):
            data_handler.get_intensity('Unknown Country')

def test_get_countries(data_handler):
    assert data_handler._get_countries() == ['Austria', 'United States']

def test_get_latest_year_by_country_code(data_handler):
    assert data_handler._get_latest_year_by_country_code('USA') == 2021
    assert data_handler._get_latest_year_by_country_code('AUT') == 2021
    assert data_handler._get_latest_year_by_country_code('ZZZ') is None
