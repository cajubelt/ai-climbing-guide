from enum import StrEnum

ELASTICSEARCH_INDEX_NAME = 'openbeta'


class ClimbStyle(StrEnum):
    trad = "trad"
    sport = "sport"
    boulder = "boulder"
    mixed = "mixed"
