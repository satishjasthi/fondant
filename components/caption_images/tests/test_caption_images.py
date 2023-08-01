import ast

import pandas as pd
import requests
from caption_images.src.main import CaptionImagesComponent
from fondant.abstract_component_test import AbstractComponentTest


class TestCaptionImagesComponent(AbstractComponentTest):
    def create_component(self):
        comp_args = self.test_config["component_arguments"]
        return CaptionImagesComponent(
            model_id=comp_args["model_id"],
            batch_size=comp_args["batch_size"],
            max_new_tokens=comp_args["max_new_tokens"],
        )

    def create_input_data(self):
        image_urls = self.test_config["input_data"]["image_urls"]
        return pd.DataFrame(
            {"images": {"data": [requests.get(url).content for url in image_urls]}},
        )

    def create_output_data(self):
        return pd.DataFrame(
            data={
                ast.literal_eval(key): {
                    int(nested_key): nested_value
                    for nested_key, nested_value in value.items()
                }
                for key, value in self.test_config["expected_output"].items()
            },
        )
