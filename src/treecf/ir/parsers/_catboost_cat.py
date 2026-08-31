"""CatBoost categorical support: category hashing and CTR lowering.

CatBoost identifies a category by ``CityHash64(utf8(value)) & 0xffffffff``
(the classic CityHash, the variant CatBoost vendors). The functions here
reproduce that hash bit-for-bit so parsed models can map the caller's category
names onto the hash values stored in the model dump; every conformance run
cross-checks the reproduction against the model's own stored values.
"""

from __future__ import annotations

MASK64 = (1 << 64) - 1

_K0 = 0xC3A5C85C97CB3127
_K1 = 0xB492B66FBE98F273
_K2 = 0x9AE16A3B2F90404F
_K3 = 0xC949D7C7509E6557
_K_MUL = 0x9DDFEA08EB382D69


def _fetch64(data: bytes, pos: int) -> int:
    return int.from_bytes(data[pos : pos + 8], "little")


def _fetch32(data: bytes, pos: int) -> int:
    return int.from_bytes(data[pos : pos + 4], "little")


def _rotate(value: int, shift: int) -> int:
    if shift == 0:
        return value
    return ((value >> shift) | (value << (64 - shift))) & MASK64


def _shift_mix(value: int) -> int:
    return (value ^ (value >> 47)) & MASK64


def _hash128_to_64(low: int, high: int) -> int:
    a = ((low ^ high) * _K_MUL) & MASK64
    a ^= a >> 47
    b = ((high ^ a) * _K_MUL) & MASK64
    b ^= b >> 47
    return (b * _K_MUL) & MASK64


def _hash_len16(u: int, v: int) -> int:
    return _hash128_to_64(u, v)


def _hash_len_0_to_16(data: bytes) -> int:
    n = len(data)
    if n > 8:
        a = _fetch64(data, 0)
        b = _fetch64(data, n - 8)
        return (_hash_len16(a, _rotate((b + n) & MASK64, n)) ^ b) & MASK64
    if n >= 4:
        a = _fetch32(data, 0)
        return _hash_len16((n + (a << 3)) & MASK64, _fetch32(data, n - 4))
    if n > 0:
        a, b, c = data[0], data[n >> 1], data[n - 1]
        y = (a + (b << 8)) & MASK64
        z = (n + (c << 2)) & MASK64
        return (_shift_mix((y * _K2 ^ z * _K3) & MASK64) * _K2) & MASK64
    return _K2


def _hash_len_17_to_32(data: bytes) -> int:
    n = len(data)
    a = (_fetch64(data, 0) * _K1) & MASK64
    b = _fetch64(data, 8)
    c = (_fetch64(data, n - 8) * _K2) & MASK64
    d = (_fetch64(data, n - 16) * _K0) & MASK64
    return _hash_len16(
        (_rotate((a - b) & MASK64, 43) + _rotate(c, 30) + d) & MASK64,
        (a + _rotate((b ^ _K3) & MASK64, 20) - c + n) & MASK64,
    )


def _hash_len_33_to_64(data: bytes) -> int:
    n = len(data)
    z = _fetch64(data, 24)
    a = (_fetch64(data, 0) + (n + _fetch64(data, n - 16)) * _K0) & MASK64
    b = _rotate((a + z) & MASK64, 52)
    c = _rotate(a, 37)
    a = (a + _fetch64(data, 8)) & MASK64
    c = (c + _rotate(a, 7)) & MASK64
    a = (a + _fetch64(data, 16)) & MASK64
    vf = (a + z) & MASK64
    vs = (b + _rotate(a, 31) + c) & MASK64
    a = (_fetch64(data, 16) + _fetch64(data, n - 32)) & MASK64
    z = _fetch64(data, n - 8)
    b = _rotate((a + z) & MASK64, 52)
    c = _rotate(a, 37)
    a = (a + _fetch64(data, n - 24)) & MASK64
    c = (c + _rotate(a, 7)) & MASK64
    a = (a + _fetch64(data, n - 16)) & MASK64
    wf = (a + z) & MASK64
    ws = (b + _rotate(a, 31) + c) & MASK64
    r = _shift_mix(((vf + ws) * _K2 + (wf + vs) * _K0) & MASK64)
    return (_shift_mix((r * _K0 + vs) & MASK64) * _K2) & MASK64


def _weak_hash_len32_with_seeds(data: bytes, pos: int, a: int, b: int) -> tuple[int, int]:
    w = _fetch64(data, pos)
    x = _fetch64(data, pos + 8)
    y = _fetch64(data, pos + 16)
    z = _fetch64(data, pos + 24)
    a = (a + w) & MASK64
    b = _rotate((b + a + z) & MASK64, 21)
    c = a
    a = (a + x) & MASK64
    a = (a + y) & MASK64
    b = (b + _rotate(a, 44)) & MASK64
    return (a + z) & MASK64, (b + c) & MASK64


def city_hash_64(data: bytes) -> int:
    """Classic CityHash64 (the pre-1.1 variant CatBoost vendors)."""
    n = len(data)
    if n <= 32:
        if n <= 16:
            return _hash_len_0_to_16(data)
        return _hash_len_17_to_32(data)
    if n <= 64:
        return _hash_len_33_to_64(data)

    x = _fetch64(data, 0)
    y = (_fetch64(data, n - 16) ^ _K1) & MASK64
    z = (_fetch64(data, n - 56) ^ _K0) & MASK64
    v = _weak_hash_len32_with_seeds(data, n - 64, n, y)
    w = _weak_hash_len32_with_seeds(data, n - 32, (n * _K1) & MASK64, _K0)
    z = (z + _shift_mix(v[1]) * _K1) & MASK64
    x = (_rotate((z + x) & MASK64, 39) * _K1) & MASK64
    y = (_rotate(y, 33) * _K1) & MASK64

    pos = 0
    remaining = (n - 1) & ~63
    while True:
        x = (_rotate((x + y + v[0] + _fetch64(data, pos + 16)) & MASK64, 37) * _K1) & MASK64
        y = (_rotate((y + v[1] + _fetch64(data, pos + 48)) & MASK64, 42) * _K1) & MASK64
        x ^= w[1]
        y ^= v[0]
        z = _rotate((z ^ w[0]) & MASK64, 33)
        v = _weak_hash_len32_with_seeds(data, pos, (v[1] * _K1) & MASK64, (x + w[0]) & MASK64)
        w = _weak_hash_len32_with_seeds(data, pos + 32, (z + w[1]) & MASK64, y)
        z, x = x, z
        pos += 64
        remaining -= 64
        if remaining == 0:
            break
    return _hash_len16(
        (_hash_len16(v[0], w[0]) + _shift_mix(y) * _K1 + z) & MASK64,
        (_hash_len16(v[1], w[1]) + x) & MASK64,
    )


def cat_feature_hash(value: str) -> int:
    """CatBoost's category identifier: the low 32 bits of CityHash64(utf8)."""
    return city_hash_64(value.encode("utf-8")) & 0xFFFFFFFF


def signed32(value: int) -> int:
    """The JSON rendering CatBoost uses for hash values (two's-complement int32)."""
    return value - (1 << 32) if value >= (1 << 31) else value


_CTR_MAGIC = 0x4906BA494954CB65


def calc_ctr_bucket(hash32: int) -> int:
    """The statistics-table bucket for one category.

    CatBoost chains its multiplicative hash over the category's identifier,
    sign-extending the 32-bit value to 64 bits first: the seen-category
    buckets in a model's table are exactly these values.
    """
    extended = signed32(hash32) & MASK64
    return (_CTR_MAGIC * ((_CTR_MAGIC * extended) & MASK64)) & MASK64
