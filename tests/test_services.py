import pytest

from app import services as sv
from app.services import ShopError
from tests.conftest import STORE

GOOD = dict(customer_name="Ayesha Khan", phone="0300 1234567", address="House 12, Street 4, G-11/2", city="Islamabad")


def test_search_filters_colour_and_budget(s):
    skus = [p["sku"] for p in sv.search_catalog(s, STORE, color="red", max_price=5000)]
    assert skus == ["NT-401", "NT-201", "NT-101"]                       # cheapest first, all <= 5000


def test_search_any_word_best_match_first(s):
    skus = [p["sku"] for p in sv.search_catalog(s, STORE, query="dupatta trouser", color="red")]
    assert "NT-401" in skus                                             # the red dupatta upsell
    assert [p["sku"] for p in sv.search_catalog(s, STORE, query="chiffon dupatta")][0] == "NT-401"


def test_search_hides_out_of_stock(s):
    assert "NT-103" not in [p["sku"] for p in sv.search_catalog(s, STORE, category="unstitched")]


def test_search_size_filter_keeps_one_size_items(s):
    skus = [p["sku"] for p in sv.search_catalog(s, STORE, size="s", category="kurta")]
    assert "NT-203" not in skus and "NT-201" in skus                    # NT-203 has no S
    assert sv.search_catalog(s, STORE, size="M", category="dupatta")    # dupattas are ONE SIZE


def test_search_never_returns_other_store(s):
    assert all(not p["sku"].startswith("OS-") for p in sv.search_catalog(s, STORE, query="secret"))
    assert sv.search_catalog(s, STORE, query="secret") == []


def test_check_stock(s):
    assert sv.check_stock(s, STORE, "nt-201", "m")["available"] == 4
    with pytest.raises(ShopError):
        sv.check_stock(s, STORE, "OS-1")                                # other store's product


def test_total_uses_db_prices_and_city_fee(s):
    q = sv.calculate_total(s, STORE, [{"sku": "NT-201", "size": "M", "qty": 2},
                                      {"sku": "NT-401", "size": "M", "qty": 1}], "rawalpindi")
    assert (q["subtotal"], q["delivery_fee"], q["total"]) == (7600, 200, 7800)
    assert q["lines"][1]["size"] == "ONE SIZE"
    assert sv.calculate_total(s, STORE, [{"sku": "NT-202", "size": "S", "qty": 1}], "Lahore")["delivery_fee"] == 350


@pytest.mark.parametrize("items", [[], [{"sku": "NT-301", "size": "M", "qty": 5}],
                                   [{"sku": "NT-202", "size": "M", "qty": 0}], [{"sku": "NOPE", "size": "M"}]])
def test_total_rejects_bad_carts(s, items):
    with pytest.raises(ShopError):
        sv.calculate_total(s, STORE, items, "Islamabad")


def test_place_order_reserves_stock_and_awaits_approval(s):
    o = sv.place_order(s, STORE, "user-a", [{"sku": "NT-201", "size": "M", "qty": 2}], **GOOD)
    assert o["status"] == "awaiting_approval" and o["total"] == 6600
    assert sv.check_stock(s, STORE, "NT-201", "M")["available"] == 2


@pytest.mark.parametrize("field,value", [("phone", "12345"), ("address", "G-11"), ("customer_name", "")])
def test_place_order_validates_customer(s, field, value):
    with pytest.raises(ShopError):
        sv.place_order(s, STORE, "user-a", [{"sku": "NT-202", "size": "M"}], **{**GOOD, field: value})


def test_customer_sees_only_own_orders(s):
    o = sv.place_order(s, STORE, "user-a", [{"sku": "NT-202", "size": "M"}], **GOOD)
    assert sv.get_orders(s, STORE, "user-a", o["order_id"])[0]["order_id"] == o["order_id"]
    with pytest.raises(ShopError):
        sv.get_orders(s, STORE, "user-b", o["order_id"])                # User B can't see A's order
    with pytest.raises(ShopError):
        sv.get_orders(s, "other-store", "user-a", o["order_id"])        # nor from another store
    assert sv.get_orders(s, STORE, "user-b") == []


def test_seller_approve_and_reject(s):
    a = sv.place_order(s, STORE, "user-a", [{"sku": "NT-302", "size": "L"}], **GOOD)
    b = sv.place_order(s, STORE, "user-a", [{"sku": "NT-204", "size": "M", "qty": 2}], **GOOD)
    assert sv.review_order(s, STORE, a["order_id"], approve=True)["status"] == "approved"
    assert sv.review_order(s, STORE, b["order_id"], approve=False)["status"] == "rejected"
    assert sv.check_stock(s, STORE, "NT-204", "M")["available"] == 2    # stock given back
    with pytest.raises(ShopError):
        sv.review_order(s, STORE, a["order_id"], approve=False)         # already decided
    with pytest.raises(ShopError):
        sv.review_order(s, "other-store", b["order_id"], approve=True)  # other seller can't touch it
