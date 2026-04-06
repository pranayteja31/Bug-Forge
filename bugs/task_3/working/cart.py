def apply_coupon(total, coupon_type):
    if coupon_type == "FLAT50":
        return total - 50
    elif coupon_type == "HALF":
        return total / 2
