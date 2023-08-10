import json
import re


class PromptInfoExtractor:
    def __init__(self, prompt, config_filepath=None):
        self.data = prompt
        #        self.load_data(json_filepath)
        if config_filepath:
            self.load_config(config_filepath)
        else:
            self.config = {
                "search_class_types": ["KSampler", "KSamplerAdvanced"],
                "output_format": "Steps: {steps}, Sampler: {sampler_name} {scheduler}, CFG scale: {cfg}, Seed: {seed}, Size: {width}x{height}, Model: {model_name}",
            }
        self.info = self.gather_info()

    def load_data(self, json_filepath):
        """Load JSON data from the provided filepath."""
        with open(json_filepath, "r") as file:
            self.data = json.load(file)

    def load_config(self, config_filepath):
        """Load configuration from the provided filepath."""
        with open(config_filepath, "r") as config_file:
            self.config = json.load(config_file)

    def gather_info(self):
        ksampler_items = self.get_ksampler_items()

        if not ksampler_items:
            return None

        key, item = ksampler_items[0]

        model_name = self.extract_model_name(item)
        latent_image_info = self.extract_latent_image_info(item)

        info_dict = {
            "steps": item["inputs"]["steps"],
            "sampler_name": item["inputs"]["sampler_name"],
            "scheduler": item["inputs"]["scheduler"],
            "cfg": item["inputs"]["cfg"],
            "seed": item["inputs"].get("seed", item["inputs"].get("noise_seed", None)),
            "width": latent_image_info["inputs"]["width"],
            "height": latent_image_info["inputs"]["height"],
            "model_name": model_name,
        }

        self.extract_prompt_info(info_dict)

        return info_dict

    def get_ksampler_items(self):
        ksampler_items = [
            (k, v)
            for k, v in self.data.items()
            if v["class_type"] in self.config["search_class_types"]
        ]
        return sorted(ksampler_items, key=lambda x: int(x[0]))

    def extract_model_name(self, item):
        return self.get_ckpt_name(item["inputs"]["model"][0]).replace("\\", "_")

    def get_ckpt_name(self, node_number):
        """Recursively search for the 'ckpt_name' key starting from the specified node."""
        node = self.data[node_number]
        if "ckpt_name" in node["inputs"]:
            return node["inputs"]["ckpt_name"]
        if "model" in node["inputs"]:
            return self.get_ckpt_name(node["inputs"]["model"][0])
        return None

    def extract_latent_image_info(self, item):
        latent_image_node_number = item["inputs"]["latent_image"][0]
        return self.data[latent_image_node_number]

    def extract_prompt_info(self, info_dict):
        positive_text = self.extract_text_by_key("positive")
        negative_text = self.extract_text_by_key("negative")

        if positive_text:
            info_dict["prompt"] = positive_text
        if negative_text:
            info_dict["negative"] = negative_text

    def extract_text_by_key(self, key):
        """Extract text by the given key, either 'positive' or 'negative'."""
        ksampler_items = self.get_ksampler_items()

        if not ksampler_items:
            return None

        ksampler_item = ksampler_items[0][1]

        if key not in ksampler_item["inputs"]:
            return None

        target_node_number = ksampler_item["inputs"][key][0]
        target_node = self.data[target_node_number]

        return target_node["inputs"]["text"]

    def format_info(self, info_dict):
        """Format the gathered information based on the configuration."""
        formatted_str = self.config["output_format"].format(**info_dict)
        return formatted_str

    def extract_and_format(self):
        """Extract and format the required information from the loaded JSON data."""
        info = self.gather_info()
        if not info:
            return "No suitable data found."
        return self.format_info(info)

    def formatted_annotation(self):
        annotation = ""
        if len(self.info["prompt"]) > 0:
            annotation += self.info["prompt"]

        if len(self.info["negative"]) > 0:
            if len(annotation) > 0:
                annotation += "\n"
            annotation += "Negative prompt: "
            annotation += self.info["negative"]

        if len(annotation) > 0:
            annotation += "\n"
        annotation += self.extract_and_format()

        return annotation

    def get_prompt_tags(self):
        cleaned_string = re.sub(r":\d+\.\d+", "", self.info["prompt"])
        items = cleaned_string.split(",")

        return [
            re.sub(r"[\(\)]", "", item).strip()
            for item in items
            if re.sub(r"[\(\)]", "", item).strip()
        ]
