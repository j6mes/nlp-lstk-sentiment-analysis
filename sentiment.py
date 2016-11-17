def fine_grained(x):
    if x <= 0.2:
        return "NegNeg"
    elif x>0.2 and x<=0.4:
        return "Neg"
    elif x>0.4 and x<=0.6:
        return "Neu"
    elif x>0.6 and x<=0.8:
        return "Pos"
    elif x>0.8:
        return "PosPos"

def polar(x):
    if x <= 0.4:
        return "Neg"
    elif x>0.6:
        return "Pos"
    else:
        return None

class_functions = [polar,fine_grained]