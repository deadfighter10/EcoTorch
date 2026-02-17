import pytest
from unittest.mock import patch, MagicMock
from ecotorch.geolocator import get_ip, get_location, convert_country_to_iso
import requests

def test_get_ip_success():
    with patch('requests.get') as mock_get:
        mock_get.return_value.text = '1.2.3.4'
        assert get_ip() == '1.2.3.4'

def test_get_ip_failure():
    with patch('requests.get', side_effect=requests.RequestException):
        with pytest.raises(requests.ConnectionError):
            get_ip()

def test_convert_country_to_iso_success():

    with patch('ecotorch.geolocator.pycountry.countries.search_fuzzy') as mock_search:
        mock_country = MagicMock()
        mock_country.alpha_3 = 'USA'
        mock_search.return_value = [mock_country]

        assert convert_country_to_iso('United States') == 'USA'

def test_convert_country_to_iso_failure():
    with patch('ecotorch.geolocator.pycountry.countries.search_fuzzy', side_effect=LookupError):
        assert convert_country_to_iso('Unknown Country') is None

def test_get_location_success():
    with patch('ecotorch.geolocator.get_ip', return_value='1.2.3.4'), \
         patch('ecotorch.geolocator.GeoIP2Fast') as mock_geoip_cls, \
         patch('ecotorch.geolocator.convert_country_to_iso', return_value='USA') as mock_convert:

        mock_geoip_instance = mock_geoip_cls.return_value
        mock_geoip_instance.lookup.return_value.country_name = 'United States'

        assert get_location() == 'USA'
        mock_convert.assert_called_with('United States')
