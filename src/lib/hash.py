# pyhash.py
# Charles J. Lai
# October 11, 2013

"""
See README file for top level documentation.
"""


# == FNV (Fowler-Noll-Vo) hashes ==========================================
def fnv1_32(string, seed=0):
    """
    Returns: The FNV-1 hash of a given string.
    """
    # Constants
    FNV_prime = 16777619
    offset_basis = 2166136261

    # FNV-1a Hash Function
    hash = offset_basis + seed
    for char in string:
        hash = hash * FNV_prime
        hash = hash ^ ord(char)
    return hash


def fnv1a_32(string, seed=0):
    """
    Returns: The FNV-1a (alternate) hash of a given string
    """
    # Constants
    FNV_prime = 16777619
    offset_basis = 2166136261

    # FNV-1a Hash Function
    hash = offset_basis + seed
    for char in string:
        hash = hash ^ ord(char)
        hash = hash * FNV_prime
    return hash


def fnv1_64(string, seed=0):
    """
    Returns: The FNV-1 hash of a given string.
    """
    # Constants
    FNV_prime = 1099511628211
    offset_basis = 14695981039346656037

    # FNV-1a Hash Function
    hash = offset_basis + seed
    for char in string:
        hash = hash * FNV_prime
        hash = hash ^ ord(char)
    return hash


def fnv1a_64(string, seed=0):
    """
    Returns: The FNV-1a (alternate) hash of a given string
    """
    # Constants
    FNV_prime = 1099511628211
    offset_basis = 14695981039346656037

    # FNV-1a Hash Function
    hash = offset_basis + seed
    for char in string:
        hash = hash ^ ord(char)
        hash = hash * FNV_prime
    return hash


# == TESTING APPLICATION ==================================================
def main():
    """
    Testing application: Do something
    """
    print(fnv1a_32("lol", 2))


if __name__ == '__main__':
    main()
