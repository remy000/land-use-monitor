import json
from datetime import date
from pathlib import Path

import pandas as pd
from pystac_client import Client

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"


def load_aoi(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def search_scenes(
        bbox:list[float],
        start:date,
        end:date,
        max_cloud:float = 30.0,
        
)-> list:
    client=Client.open(EARTH_SEARCH_URL)
    search=client.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{start.isoformat()} / {end.isoformat()}",
        query={"eo:cloud_cover": {"lt": max_cloud}}
    )
    return list(search.items())
    