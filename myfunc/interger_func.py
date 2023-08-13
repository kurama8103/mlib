

def eratosthenes(n: int) -> list:
    primes = [True]*n
    primes[0] = False
    primes[1] = False
    for i in range(2, int(n**(1/2))+1):
        if primes[i]:
            for j in range(i*2, n, i):
                primes[j] = False
    return [i for i in range(n) if primes[i]]


def euclidean_gcd(a: int, b: int) -> int:
    a, b = sorted((a, b), reverse=True)
    while b > 0:
        a, b = b, a % b
    return a


def make_divisors(n: int) -> list:
    l_div, u_div = [], []
    i = 1
    while i*i <= n:
        if n % i == 0:
            l_div.append(i)
            if i != n // i:
                u_div.append(n//i)
        i += 1
    u_div.reverse()
    return l_div + u_div
