from collections import Counter
import heapq


def merge(a: list, l: int, m: int, r: int) -> list:
    t = a[l:m] + a[m:r][::-1]
    n = len(t)
    i = 0
    j = n - 1
    b = [None] * n
    for k in range(n):
        if t[i] <= t[j]:
            b[k] = t[i]
            i += 1
        else:
            b[k] = t[j]
            j -= 1
    return b


def merge_sort(a: list, l: int = 0, r: int = -1) -> list:
    if r == -1:
        r = len(a)
    if l + 1 < r:
        m = (l + r) // 2
        a = merge_sort(a, l, m)
        a = merge_sort(a, m, r)
        a[l:r] = merge(a, l, m, r)
        print((l, m, r), a)
    return a


def partition(a: list, l: int, r: int) -> int:
    i = l - 1
    for j in range(l, r):
        if a[j] < a[r]:
            i += 1
            a[i], a[j] = a[j], a[i]

    i += 1
    a[i], a[r] = a[r], a[i]
    return i


def quick_sort(a: list, l: int = 0, r: int = -1) -> list:
    if r == -1:
        r = len(a) - 1
    if r > l:
        q = partition(a, l, r)
        print(q, a)
        quick_sort(a, l, q - 1)
        quick_sort(a, q + 1, r)
    return a


def bubble_sort(a: list) -> list:
    n = len(a)
    for i in range(n - 1):
        for j in range(n - 1, i, -1):
            if a[j - 1] > a[j]:
                a[j], a[j - 1] = a[j - 1], a[j]
                print((i, j - 1), a)
    return a


def selection_sort(a: list) -> list:
    n = len(a)
    for i in range(n - 1):
        m = idx_min(a[i:n])
        a[i], a[m + i] = a[m + i], a[i]
        print((i, m + i), a)
    return a


def idx_min(a: list) -> int:
    # return sorted(list(zip(a,range(len(a)))))[0][1]
    m = 0
    for i in range(len(a)):
        if a[i] < a[m]:
            m = i
    return m


def insert_sort(a: list, step: int = 1) -> list:
    for i in range(1, len(a), step):
        t = a.pop(i)
        for j in range(i - 1, -1, -1):
            if a[j] < t:
                j += 1
                break

        a.insert(j, t)
        print((i, j), a)
    return a


def shell_sort(a: list) -> list:
    for i in range(len(a) // 2, 0, -2):
        insert_sort(a, i)
    return a


def heap_sort(a: list) -> list:
    heapq.heapify(a)
    return [heapq.heappop(a) for _ in range(len(a))]


def counting_sort(a: list) -> list:
    c = Counter(a)
    t = 0
    for i in sorted(c.keys()):
        t += c[i]
        c[i] = t

    l = [None] * len(a)
    for i in range(len(a) - 1, -1, -1):
        x = a[i]
        c[x] -= 1
        l[c[x]] = x

    return l
