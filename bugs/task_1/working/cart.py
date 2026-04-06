from utils import calculate_discount, calculate_tax

def get_final_price(price, discount_pct, tax_rate):
    discounted = price - calculate_discount(price, discount_pct)
    final = discounted + calculate_tax(discounted, tax_rate)
    return round(final, 2)