from models import get_item_quantity

def calculate_total(item_id, price):
    qty = get_item_quantity(item_id)
    return qty * price