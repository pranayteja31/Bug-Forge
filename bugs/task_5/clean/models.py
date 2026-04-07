def get_shipping_zone(zip_code):
    if zip_code.startswith(("9", "8")):
        return "west"
    return "standard"
