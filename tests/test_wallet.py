import pytest
import json
from pathlib import Path
from unittest.mock import patch, mock_open
from ecotorch.wallet.extractor import Extractor

def test_extractor_init():
    with patch("ecotorch.wallet.extractor.Path") as mock_path:
        # Mocking the path to electricity_price.json
        mock_file_path = mock_path.return_value / "electricity_price.json"
        
        # Mocking json load
        mock_data = {"USA": 0.12, "DEU": 0.35}
        with patch("ecotorch.wallet.extractor.open", mock_open(read_data=json.dumps(mock_data))):
            extractor = Extractor()
            assert extractor.prices == mock_data

def test_extractor_calculate_cost():
    mock_data = {"USA": 0.12, "DEU": 0.35}
    with patch("ecotorch.wallet.extractor.Extractor._load_json", return_value=mock_data):
        extractor = Extractor()
        
        # USA: 0.12 * 10 = 1.2
        assert extractor.calculate_cost("USA", 10) == 1.2
        # DEU: 0.35 * 2 = 0.7
        assert extractor.calculate_cost("DEU", 2) == 0.7
        # Non-existent country should raise KeyError (based on implementation)
        with pytest.raises(KeyError):
            extractor.calculate_cost("FRA", 1)

def test_extractor_load_json_errors():
    with patch("ecotorch.wallet.extractor.open", side_effect=FileNotFoundError):
        extractor = Extractor.__new__(Extractor)
        with pytest.raises(FileNotFoundError, match="Internal error, please contact the developers."):
            extractor._load_json("fake_path")

    with patch("ecotorch.wallet.extractor.open", mock_open(read_data="invalid json")):
        extractor = Extractor.__new__(Extractor)
        with pytest.raises(json.JSONDecodeError, match="Internal error, please contact developers."):
            extractor._load_json("fake_path")
