def linear_srgb(x):
    return (x <= 0.0031308).float()*(12.92*x) + (x > 0.0031308).float()*(1.055*x**(1/2.4) - 0.055)

def linear_srgb_approx(x):
    return x**2.2

def rgb2hexstr(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)
