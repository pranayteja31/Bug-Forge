def get_item_quantity(item_id):
    quantities = {"A01": 5, "B02": 3, "C03": 8}
    return str(quantities.get(item_id, 0))