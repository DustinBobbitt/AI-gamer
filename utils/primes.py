"""
Prime number utilities for semiprime factoring.
Adapted from Prime Ice project.
"""
from __future__ import annotations

from math import isqrt
from typing import List


def is_prime(n: int) -> bool:
    """Deterministic primality check for 32-bit integers."""
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
    if n in small_primes:
        return True
    for p in small_primes:
        if n % p == 0:
            return False

    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    def check(a: int) -> bool:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                return True
        return False

    for a in (2, 7, 61):
        if a % n == 0:
            continue
        if not check(a):
            return False
    return True


def primes_up_to(limit: int) -> List[int]:
    """Return list of all primes <= limit using a simple sieve."""
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, isqrt(limit) + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start : limit + 1 : step] = b"\x00" * (((limit - start) // step) + 1)
    return [i for i, is_p in enumerate(sieve) if is_p]


def generate_prime(bit_length: int) -> int:
    """
    Generate a random prime with specified bit length.
    Uses rejection sampling with Miller-Rabin primality test.
    
    Args:
        bit_length: Number of bits in the prime
    
    Returns:
        A prime number with the specified bit length
    """
    import random
    
    if bit_length < 2:
        raise ValueError("bit_length must be >= 2")
    
    # Generate random odd number in range [2^(n-1), 2^n - 1]
    min_val = 1 << (bit_length - 1)
    max_val = (1 << bit_length) - 1
    
    # Try up to 1000 candidates (expected ~log(N) tries)
    for _ in range(1000):
        candidate = random.randrange(min_val, max_val + 1)
        if candidate % 2 == 0:
            candidate += 1
        if is_prime(candidate):
            return candidate
    
    raise RuntimeError(f"Failed to generate prime with {bit_length} bits after 1000 attempts")

