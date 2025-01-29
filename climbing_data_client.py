from abc import ABC, abstractmethod


class ClimbingDataClient(ABC):
    @abstractmethod
    def search_climbs(self, route_name):
        pass
