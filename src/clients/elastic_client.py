from clients.climbing_data_client import ClimbingDataClient
from constants import ELASTICSEARCH_INDEX_NAME, ClimbStyle
from elasticsearch import Elasticsearch
from typing import Optional, TypedDict


class Location(TypedDict):
    lat: int
    lon: int


class ElasticClient(ClimbingDataClient):
    def __init__(self, elastic_url, elastic_api_key):
        self.es = Elasticsearch(elastic_url, api_key=elastic_api_key)

    def search_climbs(
        self,
        route_name: Optional[str] = None,
        sector_name: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[Location] = None,
        location_radius_miles: Optional[int] = 50,
        style: Optional[ClimbStyle] = None,
        rating_min: Optional[float] = None,
        grades: Optional[list[str]] = None,
    ):
        if not self.es.indices.exists(index=ELASTICSEARCH_INDEX_NAME):
            raise Exception(f"Index {ELASTICSEARCH_INDEX_NAME} does not exist")
        query = {"bool": {"must": []}}
        if route_name is not None:
            query["bool"]["must"].append({
                "match": {
                    "route_name": {
                        "query": route_name,
                        "fuzziness": "AUTO",
                        "operator": "and"
                    }
                }
            })
        if description is not None:
            # TODO switch to vector search
            query["bool"]["must"].append({
                "match": {
                    "description": {
                        "query": description,
                        "fuzziness": "AUTO",
                        "operator": "and"
                    }
                }
            })
        if sector_name is not None:
            query["bool"]["must"].append({
                "match": {
                    "sector_name": {
                        "query": sector_name,
                        "fuzziness": "AUTO",
                        "operator": "and"
                    }
                }
            })
        if location is not None:
            if location_radius_miles is None:
                raise ValueError(
                    "location_radius_miles cannot be None when location is not None"
                )
            query["bool"]["must"].append({
                "geo_distance": {
                    "distance": f"{location_radius_miles}mi",
                    "location": location,
                }
            })

        if rating_min is not None:
            query["bool"]["must"].append({
                "range": {
                    "rating": {
                        "gte": rating_min
                    }
                }
            })

        if grades is not None:
            query["bool"]["must"].append({
                "terms": {
                    "grade": grades
                }
            })
        if style is not None:
            query["bool"]["must"].append({
                "match": {
                    "style": style
                }
            })
        print("query", query)
        response = self.es.search(
            index=ELASTICSEARCH_INDEX_NAME,
            query=query,
        )

        # Extract and reshape the relevant route information
        routes = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            route = {
                "route_name": source["route_name"],
                "route_id": source["route_id"],
                "sector_id": source["sector_id"],
                "sector_name": source["sector_name"],
                "grade": source["grade"],
                "style": source["style"],
                "description": source["description"],
                "rating": source["rating"],
                "location": source["location"],
                "score": hit["_score"],
            }
            routes.append(route)

        return {"total": response["hits"]["total"]["value"], "routes": routes}
