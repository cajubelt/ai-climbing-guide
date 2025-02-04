import pandas as pd
from scripts.load_climbing_data import transform_data
from unittest.mock import patch

@patch("scripts.load_climbing_data.load_dotenv")
@patch("scripts.load_climbing_data.OpenAI")
@patch("scripts.load_climbing_data.add_embeddings")
def test_transform_data(mock_add_embeddings, mock_openai, mock_load_dotenv):
    df = pd.DataFrame([{'route_name': 'Stairway to Heaven', 'parent_sector': 'Drive In Wall', 'route_ID': 106956280, 'sector_ID': '106947227', 'type_string': 'trad', 'fa': 'unknown', 'YDS': '5.7', 'Vermin': None, 'nopm_YDS': '5.7', 'nopm_Vermin': None, 'YDS_rank': 73.0, 'Vermin_rank': None, 'safety': '', 'parent_loc': [-91.5625, 42.614], 
    'description': ['Climb the large flake...'], 'location': '', 'protection': ['SR, tricams are handy.'], 
    'corrected_users_ratings': [('e9977e5af38e002307bada00a10a9e3cdd990c80', 1.0), ('a4a0781ac4f40e0fe97b6d39713d745486d91095', 3.0)]}])
    transformed_data = transform_data(df)
    assert transformed_data[0] == {
        "route_name": "Stairway to Heaven",
        "route_id": 106956280,
        "sector_id": "106947227",
        "grade": "5.7",
        "sector_name": "Drive In Wall",
        "location": {"lat": 42.614, "lon": -91.5625},
        "style": "trad",
        "description": "Climb the large flake...",
        "description_vector": None,
        "rating": 2.0
    }