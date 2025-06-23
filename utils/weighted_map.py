import random
import time

from pydantic import BaseModel

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class WeightedValue(BaseModel):
    """A value to represent values within a WeightedMap."""

    value: object
    weight: float


class WeightedMap:
    """A new data structure for managing multiple values with varying weights."""

    def __init__(self, mapping: list[WeightedValue]):
        """Init the map."""
        self.listing = []
        self.corresponding_weights = []
        for element in mapping:
            self.listing.append(element.value)
            self.corresponding_weights.append(element.weight)
        self.deactivated_urls = []
        self.indices = list(range(len(mapping)))
        self.num_endpoints = len(self.listing)

        # time stamps for when each URL rate limits
        self.rate_limit_times = {idx: 0.0 for idx in self.indices}

        # required wait times for each URL after a rate limit
        self.wait_times = {idx: 0 for idx in self.indices}

    def get_random_index(self):
        """Gets a valid URL from the map based on the corresponding weights."""
        valid_urls, valid_weights = self.get_valid()
        choice = random.choices(valid_urls, weights=valid_weights, k=1)[0]
        return choice

    def get_valid(self) -> tuple[list, list]:
        """Returns the valid URLs based on their last rate limit time and required wait time. Returns list of those URLS along with their corresponding weights."""
        for url_idx in self.deactivated_urls:
            current_time = time.time()
            last_rate_limited_time = self.rate_limit_times[url_idx]
            required_wait = self.wait_times[url_idx]
            if last_rate_limited_time != 0:
                if current_time - last_rate_limited_time > required_wait:
                    self.deactivated_urls.remove(url_idx)
                    self.rate_limit_times[url_idx] = 0
                    self.wait_times[url_idx] = 0

        # edge case where all urls are deactivated
        if sorted(self.deactivated_urls) == self.indices:
            self.rate_limit_times = {key: 0 for key in self.deactivated_urls}
            self.deactivated_urls = []

        valid = []
        weights = []
        for url in self.indices:
            if url not in self.deactivated_urls:
                valid.append(url)
                weights.append(self.corresponding_weights[url])

        return valid, weights
