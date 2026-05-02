def match_person(detected_attrs, query_attrs, threshold=0.5, frame=None, bbox=None):
    """
    Match a detected person against query criteria using color attributes and keyword matching.
    Returns: (matched: bool, score: float, details: dict)
    """
    if not query_attrs:
        return False, 0.0, {}

    # 1. Color attribute matching
    clean_query = {k: v for k, v in query_attrs.items()
                   if not str(k).startswith("_") and k != "query"}
    attr_hits = 0
    matched_parts = {}

    if clean_query:
        for region, color in clean_query.items():
            detected = detected_attrs.get(region)
            is_match = (detected == color)
            matched_parts[region] = {
                "expected": color,
                "detected": detected,
                "match": is_match
            }
            if is_match:
                attr_hits += 1
        attr_score = attr_hits / len(clean_query)
        return attr_score >= threshold, attr_score, matched_parts

    # 2. Keyword fallback matching
    if "query" in query_attrs:
        q = query_attrs["query"].lower()
        d = str(detected_attrs).lower()
        match = any(w in d for w in q.split() if len(w) > 3)
        return match, 0.5 if match else 0.0, {
            "Keyword Match": {"text": q, "match": match}
        }

    return False, 0.0, {}
