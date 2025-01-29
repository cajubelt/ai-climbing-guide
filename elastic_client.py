from climbing_data_client import ClimbingDataClient
from constants import ELASTICSEARCH_INDEX_NAME
from elasticsearch import Elasticsearch


class ElasticClient(ClimbingDataClient):
    def __init__(self, elastic_url, elastic_api_key):
        self.es = Elasticsearch(elastic_url, api_key=elastic_api_key)

    def search_climbs(self, route_name):
        if not self.es.indices.exists(index=ELASTICSEARCH_INDEX_NAME):
            raise Exception(f"Index {ELASTICSEARCH_INDEX_NAME} does not exist")
        
        response = self.es.search(index=ELASTICSEARCH_INDEX_NAME, query={
            "match": {
                "route_name": {
                    "query": route_name,
                    "fuzziness": "AUTO",
                    "operator": "and"
                }
            }
        })
        
        # Extract and reshape the relevant route information
        routes = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            route = {
                'route_name': source['route_name'],
                'route_id': source['route_id'],
                'sector_id': source['sector_id'],
                'sector_name': source['sector_name'],
                'grade': source['grade'],
                'style': source['style'],
                'description': source['description'],
                'score': hit['_score']
            }
            routes.append(route)
        
        return {
            'total': response['hits']['total']['value'],
            'routes': routes
        }
