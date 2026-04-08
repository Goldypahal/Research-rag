FIGURE_TERMS = [
    "figure",
    "fig",
    "chart",
    "graph",
    "plot",
    "diagram",
    "illustration",
    "visualization"
]

def is_figure_question(query: str) -> bool:
    """
    Check if the user query is likely asking about a visual element.
    """
    query = query.lower()
    return any(term in query for term in FIGURE_TERMS)
