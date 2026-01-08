import json

def load_fashionpedia_annotations(path):
    """
    Loads Fashionpedia annotations and builds useful lookup maps.

    Returns:
    1. file_to_image_id:
       Maps image filename -> image_id
       Example: "abc.jpg" -> 3020

    2. image_id_to_attrs:
       Maps image_id -> list of attribute_ids
       Example: 3020 -> [106, 114, 127, ...]
    """
    # Load JSON annotation file
    with open(path, "r") as f:
        data = json.load(f)

    # Map: image_id -> attribute_ids
    image_id_to_attrs = {}
    for ann in data["annotations"]:
        image_id_to_attrs[ann["image_id"]] = ann["attribute_ids"]

    # Map: file_name -> image_id
    file_to_image_id = {}
    for img in data["images"]:
        file_to_image_id[img["file_name"]] = img["id"]

    return file_to_image_id, image_id_to_attrs
