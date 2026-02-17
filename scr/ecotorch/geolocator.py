
import requests
import warnings
import pycountry
from geoip2fast import GeoIP2Fast

def get_ip() -> str:
	try:
		public_ip = requests.get('https://api.ipify.org').text
	except requests.RequestException:
		raise requests.ConnectionError("Cannot find public IP, manually determine a country.")
	return public_ip

def get_location() -> str:
	warnings.filterwarnings('ignore')
	geoip = GeoIP2Fast()
	return convert_country_to_iso(geoip.lookup(get_ip()).country_name)

def convert_country_to_iso(country_name: str):
	try:
		result = pycountry.countries.search_fuzzy(country_name)[0]
		return result.alpha_3
	except (LookupError, IndexError):
		return None