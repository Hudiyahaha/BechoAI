from dataclasses import dataclass


@dataclass
class ShopContext:
    """Set by the server for every request (from the store URL and the logged-in customer).
    Tools read these, so the LLM can never choose another store or another customer."""
    store_id: str
    user_id: str
