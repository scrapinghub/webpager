import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def levenshtein_distance(s1, s2):

    cdef n1 = len(s1) + 1
    cdef n2 = len(s2) + 1

    m = [[0 for _ in range(n2)] for _ in range(n1)]

    for i from 0 <= i < n1:
        m[i][0] = i

    for j from 0 <= j < n2:
        m[0][j] = j

    for j from 1 <= j < n2:
        for i from 1 <= i < n1:
            if s1[i-1] == s2[j-1]:
                m[i][j] = m[i-1][j-1]
            else:
                m[i][j] = min(m[i-1][j] + 1, m[i][j-1] + 1, m[i-1][j-1] + 1)

    return m[n1-1][n2-1]
