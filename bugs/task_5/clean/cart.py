from models import get_shipping_zone


def estimate_shipping(zip_code, weight):
    zone = get_shipping_zone(zip_code)
    rates = {"west": 8.0, "standard": 5.0}
    return rates[zone] + (weight * 0.5)
