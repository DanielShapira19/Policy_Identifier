import requests
import os
import logging

# Initialize logger for this utility module
logger = logging.getLogger(__name__)


def web_search_validation(query: str) -> str:
    """
    Searches the web to validate healthcare billing rules and industry standards via Serper API.
    """
    logger.info(f"Initiating web search validation for query: {query[:50]}...")

    url = "https://google.serper.dev/search"
    # Focus search on US healthcare billing standards for consistency
    payload = {
        "q": f"US healthcare billing industry standard rule: {query}",
        "gl": "us",
        "hl": "en"
    }

    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        logger.error("SERPER_API_KEY not found in environment variables.")
        return "Search failed: Missing API Key."

    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }

    try:
        # Perform the external POST request
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        results = response.json()

        # Extract snippets from the first 3 organic results for context
        organic_results = results.get('organic', [])
        logger.info(f"Search successful. Found {len(organic_results)} results.")

        snippets = [res.get('snippet', '') for res in organic_results[:3]]

        if not snippets:
            logger.warning(f"No organic snippets found for query: {query}")
            return "No industry data found."

        return " ".join(snippets)

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred during web search: {http_err}")
        return f"Search failed due to HTTP error: {http_err}"
    except Exception as e:
        logger.error(f"Unexpected error during web search: {e}", exc_info=True)
        return f"Search failed: {str(e)}"