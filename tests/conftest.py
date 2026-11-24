import pytest

from app import db
from app.seed import DEMO_STORE, seed

# A second store that exists only in tests, to prove stores can't see each other's data.
OTHER_STORE = {"id": "other-store", "name": "Other Store", "delivery_fees": {}, "default_delivery_fee": 300}
OTHER_PRODUCTS = [("OS-1", "Secret Red Kurta", "kurta", "red", 1000, {"M": 9}, "Belongs to another seller.")]


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    db.configure(f"sqlite:///{tmp_path / 'test.db'}")
    seed()
    seed(OTHER_STORE, OTHER_PRODUCTS)
    yield
    db.engine.dispose()


@pytest.fixture
def s():
    with db.session() as session:
        yield session


STORE = DEMO_STORE["id"]
