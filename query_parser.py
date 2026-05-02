import re

COLOR_NAMES = {
    "red":    ([0,100,100],  [10,255,255]),
    "orange": ([11,100,100], [25,255,255]),
    "yellow": ([26,100,100], [34,255,255]),
    "green":  ([35,50,50],   [85,255,255]),
    "blue":   ([100,100,50], [130,255,255]),
    "purple": ([130,50,50],  [160,255,255]),
    "pink":   ([160,50,100], [175,255,255]),
    "white":  ([0,0,200],    [180,30,255]),
    "black":  ([0,0,0],      [180,255,50]),
    "gray":   ([0,0,50],     [180,30,200]),
    "brown":  ([10,50,50],   [20,200,150]),
}

CLOTHING_PARTS = {
    "hat": "head", "cap": "head", "helmet": "head",
    "shirt": "upper", "tshirt": "upper", "jacket": "upper",
    "hoodie": "upper", "top": "upper", "coat": "upper",
    "pants": "lower", "jeans": "lower", "shorts": "lower",
    "trousers": "lower",
    "shoes": "feet", "boots": "feet",
    "backpack": "upper", "bag": "upper",
}

VEHICLE_KEYWORDS = ["car", "bike", "motorcycle", "bus", "truck", "cycle", "bicycle", "vehicle", "scooter", "van"]

def parse_query(text):
    text = text.lower()
    query_attrs = {}

    # Detect vehicle intent
    is_vehicle_search = any(v in text for v in VEHICLE_KEYWORDS)
    if is_vehicle_search:
        query_attrs["_target_class"] = "vehicle"

    # Extract behavioral rules
    rules = []
    if "run" in text or "fast" in text: rules.append("running")
    if "loiter" in text or "wait" in text or "stand" in text: rules.append("loitering")
    if "crowd" in text or "many" in text: rules.append("crowd")
    if rules:
        query_attrs["_rules"] = rules

    # Extract color-clothing pairs
    words = re.findall(r'\w+', text)
    i = 0
    while i < len(words):
        word = words[i]
        if word in COLOR_NAMES:
            color = word
            if i + 1 < len(words):
                next_word = words[i + 1]
                if next_word in CLOTHING_PARTS:
                    region = CLOTHING_PARTS[next_word]
                    query_attrs[region] = color
                    i += 2
                    continue
        i += 1

    # Fallback to general text query
    if not query_attrs or (len(query_attrs) == 1 and "_target_class" in query_attrs):
        query_attrs["query"] = text

    return query_attrs
