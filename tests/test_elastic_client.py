import pytest
from unittest.mock import Mock, patch
from elastic_client import ElasticClient, ELASTICSEARCH_INDEX_NAME

@pytest.fixture
def mock_es_response():
    return {
        "took": 5,
        "timed_out": False,
        "_shards": {
            "total": 1,
            "successful": 1,
            "skipped": 0,
            "failed": 0
        },
        "hits": {
            "total": {
                "value": 2,
                "relation": "eq"
            },
            "max_score": 12.844319,
            "hits": [
                {
                    "_index": ELASTICSEARCH_INDEX_NAME,
                    "_id": "1",
                    "_score": 12.844319,
                    "_source": {
                        "route_name": "Transgression",
                        "route_id": 105757642,
                        "sector_id": "106243028",
                        "sector_name": "Hole in the Wall",
                        "grade": "5.10b",
                        "style": "trad",
                        "description": "Classic crack climb",
                    }
                },
                {
                    "_index": ELASTICSEARCH_INDEX_NAME,
                    "_id": "2",
                    "_score": 10.123456,
                    "_source": {
                        "route_name": "Progression",
                        "route_id": 105757643,
                        "sector_id": "106243029",
                        "sector_name": "Main Wall",
                        "grade": "5.11a",
                        "style": "sport",
                        "description": "Steep face climbing",
                    }
                }
            ]
        }
    }

@pytest.fixture
def mock_elastic_client():
    with patch('elasticsearch.Elasticsearch') as mock_es:
        es_client = Mock()
        es_client.indices.exists.return_value = True
        mock_es.return_value = es_client
        client = ElasticClient(elastic_url="http://localhost:9200", elastic_api_key="test-key")
        client.es = es_client
        yield client

def test_search_climbs_success(mock_elastic_client, mock_es_response):
    mock_elastic_client.es.search.return_value = mock_es_response
    
    result = mock_elastic_client.search_climbs("Transgression")
    
    mock_elastic_client.es.search.assert_called_once_with(
        index=ELASTICSEARCH_INDEX_NAME,
        query={"fuzzy": {"route_name": "Transgression"}}
    )
    
    assert result["total"] == 2
    assert len(result["routes"]) == 2
    
    first_route = result["routes"][0]
    assert first_route["route_name"] == "Transgression"
    assert first_route["route_id"] == 105757642
    assert first_route["sector_name"] == "Hole in the Wall"
    assert first_route["grade"] == "5.10b"
    assert first_route["style"] == "trad"
    assert first_route["score"] == 12.844319
    
    second_route = result["routes"][1]
    assert second_route["route_name"] == "Progression"
    assert second_route["route_id"] == 105757643
    assert second_route["sector_name"] == "Main Wall"
    assert second_route["grade"] == "5.11a"
    assert second_route["style"] == "sport"
    assert second_route["score"] == 10.123456

def test_search_climbs_index_not_exists(mock_elastic_client):
    mock_elastic_client.es.indices.exists.return_value = False
    
    with pytest.raises(Exception) as exc_info:
        mock_elastic_client.search_climbs("Transgression")
    
    assert str(exc_info.value) == f"Index {ELASTICSEARCH_INDEX_NAME} does not exist"
    mock_elastic_client.es.search.assert_not_called()
