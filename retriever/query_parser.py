

def parse_query(query: str):
    """
    Lightweight query intent extractor.

    Purpose:
    --------
    This function performs a minimal and interpretable parsing of the
    natural language query to extract *weak semantic intent* such as
    color cues and contextual hints (e.g., formal, casual, office).

    Important:
    ----------
    - This parser is NOT used for hard filtering.
    - It does NOT decide which images are relevant.
    - It is only used for *soft re-ranking* after CLIP-based retrieval.

    Parameters:
    -----------
    query : str
        Natural language search query provided by the user.

    Returns:
    --------
    dict
        A dictionary containing:
        - "colors": list of color keywords detected in the query
        - "context": list of contextual or style-related keywords detected

    Example:
    --------
    Input:
        "A red tie in a formal office setting"

    Output:
        {
            "colors": ["red"],
            "context": ["formal", "office"]
        }
    """
    query = query.lower()

    colors = [
        "red", "blue", "yellow", "white", "black",
        "green", "brown", "purple", "pink", "orange"
    ]

    context = [
        "office", "park", "street", "city", "home",
        "formal", "casual", "business", "party"
    ]

    return {
        "colors": [c for c in colors if c in query],
        "context": [c for c in context if c in query]
    }
