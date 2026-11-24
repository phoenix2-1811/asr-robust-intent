import re
import unicodedata

_WS = re.compile(r"\s+")


def normalize_text(s):
    s = unicodedata.normalize("NFKC", s or "").casefold()
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    out = []
    for ch in s:
        if ch == "'" or ch.isspace():
            out.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "S") or cat == "Cc":
            out.append(" ")
        else:
            out.append(ch)
    s = _WS.sub(" ", "".join(out)).strip()
    words = [w.strip("'") for w in s.split(" ")]
    return " ".join(w for w in words if w)


def word_edit_ops(ref, hyp):
    r = ref.split()
    h = hyp.split()
    n, m = len(r), len(h)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if r[i - 1] == h[j - 1] else 1
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + c)
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (0 if r[i - 1] == h[j - 1] else 1):
            ops.append(("ok" if r[i - 1] == h[j - 1] else "sub", r[i - 1], h[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append(("del", r[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, h[j - 1]))
            j -= 1
    return ops[::-1]


def word_edit_distance(ref, hyp):
    r = ref.split()
    h = hyp.split()
    n, m = len(r), len(h)
    if n == 0:
        return m, 0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m], n


def utterance_wer(ref, hyp):
    e, n = word_edit_distance(ref, hyp)
    if n == 0:
        return float(e > 0)
    return e / n


def corpus_wer(refs, hyps):
    tot_e, tot_n = 0, 0
    for r, h in zip(refs, hyps):
        e, n = word_edit_distance(r, h)
        tot_e += e
        tot_n += n
    return tot_e / max(tot_n, 1)
