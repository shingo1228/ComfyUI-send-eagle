import os
import sys
import json
import numpy as np

from PIL import Image
from datetime import datetime

import folder_paths

sys.path.append(os.path.dirname(__file__))
from eagleapi import api_item
from prompt_info_extractor import PromptInfoExtractor


class SendEagle:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "lossless_webp": (["lossy", "lossless"],),
                "compression": (
                    "INT",
                    {"default": 80, "min": 1, "max": 100, "step": 1},
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "add_item"
    OUTPUT_NODE = True
    CATEGORY = "EagleTools"

    def add_item(
        self,
        images,
        compression=80,
        lossless_webp="lossy",
        prompt=None,
        extra_pnginfo=None,
    ):
        # prompt parsing
        gen_data = PromptInfoExtractor(prompt)
        Eagle_annotation_txt = gen_data.formatted_annotation()
        Eagle_tags = gen_data.get_prompt_tags()

        subfolder_name = datetime.now().strftime("%Y-%m-%d")
        full_output_folder = os.path.join(self.output_dir, subfolder_name)

        if not os.path.exists(full_output_folder):
            os.makedirs(full_output_folder)

        lossless = lossless_webp == "lossless"

        results = list()
        for image in images:
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # get the (empty) Exif data of the generated Picture
            emptyExifData = img.getexif()
            imgexif = util.getExifFromPrompt(emptyExifData, prompt, extra_pnginfo)

            fn_modelname, _ = os.path.splitext(gen_data.info["model_name"])
            fn_num_of_smp = gen_data.info["steps"]
            fn_seed = gen_data.info["seed"]
            fn_width = gen_data.info["width"]
            fn_height = gen_data.info["height"]

            filename = f"{util.getMsecFilenameSuffix()}-{fn_modelname}-Smp-{fn_num_of_smp}-{fn_seed}-{fn_width}-{fn_height}.webp"
            filefullpath = os.path.join(full_output_folder, filename)

            img.save(filefullpath, quality=compression, exif=imgexif, lossless=lossless)

            item = api_item.EAGLE_ITEM_PATH(
                filefullpath=filefullpath,
                filename=filename,
                annotation=Eagle_annotation_txt,
                tags=Eagle_tags,
            )
            api_item.add_from_path(item=item)

            results.append(
                {"filename": filename, "subfolder": subfolder_name, "type": self.type}
            )

        return {"ui": {"images": results}}


class util:
    @classmethod
    def getMsecFilenameSuffix(cls):
        now = datetime.now()
        date_time_str = now.strftime("%Y%m%d_%H%M%S")
        return f"{date_time_str}_{now.microsecond:06}"

    @classmethod
    def getExifFromPrompt(cls, emptyExifData, prompt, extra_pnginfo):
        """Generate exif information for webp format from hidden items "prompt" and "extra_pnginfo"""
        workflowmetadata = str()
        promptstr = str()

        imgexif = emptyExifData
        if prompt is not None:
            promptstr = "".join(json.dumps(prompt))  # prepare prompt String
            imgexif[0x010F] = (
                "Prompt:" + promptstr
            )  # Add PromptString to EXIF position 0x010f (Exif.Image.Make)
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                workflowmetadata += "".join(json.dumps(extra_pnginfo[x]))
        imgexif[0x010E] = (
            "Workflow:" + workflowmetadata
        )  # Add Workflowstring to EXIF position 0x010e (Exif.Image.ImageDescription)
        return imgexif


NODE_CLASS_MAPPINGS = {
    "Send Webp Image to Eagle": SendEagle,
}
