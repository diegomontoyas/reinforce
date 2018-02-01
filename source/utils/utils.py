
def interpolate(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    # Figure out how 'wide' each range is
    left_span = in_max - in_min
    right_span = out_max - out_min

    # Convert the left range into a 0-1 range (float)
    value_scaled = float(value - in_min) / float(left_span)

    # Convert the 0-1 range into a value in the right range.
    return out_min + (value_scaled * right_span)


def normalize_0_1(value: float, in_min: float, in_max: float) -> float:
    return interpolate(value, in_min, in_max, out_min=0, out_max=1)


def get_options(argv):
    """Collect command-line options in a dictionary"""

    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts
