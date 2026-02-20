import pytest
from unittest.mock import patch, MagicMock
import requests
import ecotorch._geolocator as geolocator

def test_get_ip_success():
    with patch('requests.get') as mock_get:
        mock_get.return_value.text = '1.2.3.4'
        assert geolocator.get_ip() == '1.2.3.4'

def test_get_ip_failure():
    with patch('requests.get', side_effect=requests.RequestException):
        with pytest.raises(requests.ConnectionError):
            geolocator.get_ip()

def test_convert_country_to_iso_success():
    with patch('ecotorch._geolocator.pycountry.countries.search_fuzzy') as mock_search:
        mock_country = MagicMock()
        mock_country.alpha_3 = 'USA'
        mock_search.return_value = [mock_country]

        assert geolocator.convert_country_to_iso('United States') == 'USA'

def test_convert_country_to_iso_failure():
    with patch('ecotorch._geolocator.pycountry.countries.search_fuzzy', side_effect=LookupError):
        assert geolocator.convert_country_to_iso('Unknown Country') == 'World'

def test_get_location_success():
    with patch('ecotorch._geolocator.get_ip', return_value='1.2.3.4'), \
         patch('ecotorch._geolocator.GeoIP2Fast') as mock_geoip_cls, \
         patch('ecotorch._geolocator.convert_country_to_iso', return_value='USA') as mock_convert:

        mock_geoip_instance = mock_geoip_cls.return_value
        mock_geoip_instance.lookup.return_value.country_name = 'United States'

        assert geolocator.get_location() == 'USA'
        mock_convert.assert_called_with('United States')



def test_get_location_requests_failure():
    # If underlying IP or lookup fails with requests.RequestException, we fall back to 'World'
    with patch('ecotorch._geolocator.get_ip', side_effect=requests.RequestException):
        assert geolocator.get_location() == 'World'
