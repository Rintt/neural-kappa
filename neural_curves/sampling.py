def halton(i, base):
    # From https://sci-hub.ru/https://doi.org/10.1080/10867651.1997.10487471
    p = base
    k = i
    phi = 0.0
    while k > 0:
        a = k % base
        phi += a / p
        k = int(k / base)
        p = p*base
    return phi

class HaltonSampler:
    name = "halton"

    def __init__(self):
        self.index = 0

    def next(self):
        current_index = self.index
        self.index = (self.index + 1) # FIXME: Avoid overflow for large integers
        return halton(current_index, base=2)