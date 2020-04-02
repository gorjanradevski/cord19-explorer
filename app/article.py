from typing import Dict, Any
import numpy as np


class Article:
    def __init__(self, article_json: Dict[str, Any], query: np.ndarray):
        self.coords = np.array(article_json["coords"])
        self.id = article_json["id"]
        self.title = article_json["title"]
        self.abstract = article_json["abstract"]
        self.authors = article_json["author"]
        self.link = article_json["link"]
        self.distance_to_query = self.set_distance_to_query(query)

    def set_distance_to_query(self, query_coords: np.ndarray):
        return np.linalg.norm(self.coords - query_coords)
