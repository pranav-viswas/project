import re
from query_parser import parse_query

class NLPEngine:
    def interpret_query(self, query_text):
        query_text = query_text.lower()

        rules = []
        if any(w in query_text for w in ["run", "fast", "sprint"]): rules.append("running")
        if any(w in query_text for w in ["loiter", "wait", "stand", "linger"]): rules.append("loitering")
        if any(w in query_text for w in ["crowd", "many", "group", "gather"]): rules.append("crowd")

        spatial = None
        if "top" in query_text or "upper" in query_text: spatial = "top"
        elif "bottom" in query_text or "lower" in query_text: spatial = "bottom"
        elif "left" in query_text: spatial = "left"
        elif "right" in query_text: spatial = "right"
        elif "center" in query_text or "middle" in query_text: spatial = "center"

        alert_enabled = any(w in query_text for w in ["alert", "warn", "notify", "emergency"])
        is_counting = any(w in query_text for w in ["count", "how many", "number of", "total"])

        stop_words = ["alert", "me", "if", "is", "a", "when", "the", "to", "and",
                      "count", "find", "search", "total", "ping", "notify"]
        behavior_words = ["running", "loitering", "crowd", "erratic", "suspicious", "waiting", "standing"]
        spatial_words = ["top", "upper", "bottom", "lower", "left", "right", "center", "middle", "near"]

        target_words = [w for w in query_text.split()
                        if w not in stop_words and w not in behavior_words and w not in spatial_words]
        target_description = " ".join(target_words)

        target_class = "person"
        if "car" in query_text or "vehicle" in query_text: target_class = "car"
        elif "bag" in query_text or "backpack" in query_text: target_class = "backpack"
        elif "bike" in query_text or "motorcycle" in query_text: target_class = "motorcycle"
        elif "truck" in query_text: target_class = "truck"
        elif "bus" in query_text: target_class = "bus"
        elif "dog" in query_text: target_class = "dog"
        elif "bicycle" in query_text: target_class = "bicycle"

        return {
            "raw_query": query_text,
            "target_description": target_description or query_text,
            "target_class": target_class,
            "rules": rules,
            "spatial": spatial,
            "alert_enabled": alert_enabled,
            "is_counting": is_counting,
            "structured_attrs": parse_query(target_description)
        }

nlp = NLPEngine()
