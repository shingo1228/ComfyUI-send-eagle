import os
import sys
import tempfile
import json
import re
import numpy as np

from PIL import Image
from datetime import datetime

import folder_paths

sys.path.append(os.path.dirname(__file__))
from eagleapi import api_item

class EaglePngInfo:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
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
    CATEGORY = "EaglePngInfo"

    def add_item(
        self,
        images,
        filename_prefix="ComfyUI",
        compression=80,
        lossless_webp="lossy",
        prompt=None,
        extra_pnginfo=None,
    ):
        (
            full_output_folder,
            filename,
            counter,
            subfolder,
            filename_prefix,
        ) = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )

        lossless = lossless_webp == "lossless"

        # prompt parsing
        gen_prompt = util.classify_text_from_json(prompt)
        gen_params = util.extract_ksampler_info(prompt)

        annotation_txt = util.annotation_formatter(
            gen_prompt["Prompt"], gen_prompt["Negative"], gen_params
        )
        Eagle_tags = util.prompt2tags(gen_prompt["Prompt"])

        results = list()
        for image in images:
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # get the (empty) Exif data of the generated Picture
            emptyExifData = img.getexif()
            imgexif = util.getExifFromPrompt(emptyExifData, prompt, extra_pnginfo)

            file = f"{filename}_{util.getMsecFilenameSuffix()}.webp"
            fullfn = os.path.join(full_output_folder, file)

            img.save(fullfn, quality=compression, exif=imgexif, lossless=lossless)

            item = api_item.EAGLE_ITEM_PATH(
                filefullpath=fullfn,
                filename=file,
                annotation=annotation_txt,
                tags=Eagle_tags,
            )
            api_item.add_from_path(item=item)

            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )
            counter += 1

        return {"ui": {"images": results}}


class util:
    @classmethod
    def getMsecFilenameSuffix(cls):
        now = datetime.now()
        date_time_str = now.strftime("%Y%m%d_%H%M%S")
        return f"{date_time_str}_{now.microsecond:06}"

    @classmethod
    def getExifFromPrompt(cls, emptyExifData, prompt, extra_pnginfo):
        """webp形式用exif情報を隠し項目 "prompt", "extra_pnginfo"から生成する

        Args:
            emptyExifData (_type_): _description_
            prompt (_type_): _description_
            extra_pnginfo (_type_): _description_
        """
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

    @classmethod
    def annotation_formatter(
        cls, positive_prompt="", negative_prompt="", other_text=""
    ):
        annotation = ""
        if len(positive_prompt) > 0:
            annotation += positive_prompt

        if len(negative_prompt) > 0:
            if len(annotation) > 0:
                annotation += "\n"
            annotation += "Negative prompt: "
            annotation += negative_prompt

        if len(other_text) > 0:
            if len(annotation) > 0:
                annotation += "\n"
            annotation += other_text

        return annotation

    @classmethod
    def extract_ksampler_info(cls, prompt):
        # 2. "class_type"が"KSampler"のアイテムを検索する
        ksampler_items = {
            k: v for k, v in prompt.items() if v["class_type"] == "KSampler"
        }

        # トップノード番号が一番小さいノードのみを選択
        min_key = min(ksampler_items.keys(), key=int)
        item = ksampler_items[min_key]

        # 3. 指定された属性を取得する
        steps = item["inputs"]["steps"]
        sampler_name = item["inputs"]["sampler_name"]
        scheduler = item["inputs"]["scheduler"]
        cfg = item["inputs"]["cfg"]
        seed = item["inputs"]["seed"]

        # 4. 指定された書式で文字列を生成する
        result_str = f"Steps: {steps}, Sampler: {sampler_name} {scheduler}, CFG scale: {cfg}, Seed: {seed}"

        # 5. 生成された文字列を返す
        return result_str

    @classmethod
    def classify_text_from_json(cls, prompt):
        # Obtains the "text" attribute of an object whose "class_type" is "CLIPTextEncode"
        clip_text_encode_items = {
            k: v for k, v in prompt.items() if v["class_type"] == "CLIPTextEncode"
        }

        # Get "positive" and "negative" attributes for "class_type" of "KSampler"
        positive_keys = []
        negative_keys = []
        for item in prompt.values():
            if item["class_type"] == "KSampler":
                if "positive" in item["inputs"]:
                    positive_keys.append(item["inputs"]["positive"][0])
                if "negative" in item["inputs"]:
                    negative_keys.append(item["inputs"]["negative"][0])

        # Obtains the corresponding text attribute and determines whether it is "positive" or "negative"
        output = {}
        for key in positive_keys:
            output["Prompt"] = clip_text_encode_items[key]["inputs"]["text"]
        for key in negative_keys:
            output["Negative"] = clip_text_encode_items[key]["inputs"]["text"]

        return output

    @classmethod
    def prompt2tags(cls, csv_string):
        """生成プロンプトからキーワードを抽出しlist形式で返却

        Args:
            csv_string (_type_): 生成プロンプト

        Returns:
            _type_: タグ情報リスト(list形式)
        """
        # ':'と一緒に表記される数値を除去
        cleaned_string = re.sub(r":\d+\.\d+", "", csv_string)

        # カンマとスペースで分割
        items = cleaned_string.split(",")

        # 各項目から余分な空白や括弧を除去し、空白のみの項目を取り除く
        return [
            re.sub(r"[\(\)]", "", item).strip()
            for item in items
            if re.sub(r"[\(\)]", "", item).strip()
        ]


NODE_CLASS_MAPPINGS = {
    "Send Webp Image to Eagle": EaglePngInfo,
}
