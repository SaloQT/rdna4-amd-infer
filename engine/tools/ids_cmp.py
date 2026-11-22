"""ids_cmp.py -- compare the generated ids of ie-run logs (the 'ids:' line: prompt ids, then generated).
   python tools/ids_cmp.py N_PROMPT a.txt b.txt [c.txt ...]   (first file = reference)
Prints, per other file, the first divergence in the generated part and how many generated ids differ (after the first
divergence the texts differ, so only the first index is a clean measure)."""
import sys


def ids(path):
    for line in open(path, encoding='utf-8', errors='replace'):
        if line.startswith('ids:'):
            return [int(x) for x in line[4:].split()]
    sys.exit('no ids line in ' + path)


def main():
    n = int(sys.argv[1])
    ref = ids(sys.argv[2])
    print('reference %s: %d ids, %d generated' % (sys.argv[2], len(ref), len(ref) - n))
    for p in sys.argv[3:]:
        x = ids(p)
        d = next((i for i in range(min(len(x), len(ref))) if x[i] != ref[i]), None)
        same_after = sum(1 for i in range(n, min(len(x), len(ref))) if x[i] == ref[i])
        print('%s: %s; positions equal in the generated part: %d / %d' % (
            p, 'identical' if d is None and len(x) == len(ref) else 'first divergence at generated token %s' % (d - n if d is not None else '?'),
            same_after, len(ref) - n))


main()
