"""Codec speed on a real 151 MB linear-attention state (python-zstandard / python-lz4 are thin C wrappers; the numbers are
the C libraries' speeds on this machine, minus python call overhead which is negligible at MB sizes).

usage: python dcache_compress_speed.py FILE.iec
Prints compress and decompress MB/s (of the buffer fed to the codec) single-threaded and with N independent frames in a
thread pool (the decode layout a loader would use: one frame per layer, or per 1 MB).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import zstandard as zstd
import lz4.block
import dcache_compress_common as C


def timeit(fn, reps=3):
    best = 1e9
    out = None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


def chunks(a, n):
    k = len(a) // n
    return [a[i * k:(i + 1) * k] for i in range(n)]


def bench_zstd(name, buf, level, nchunks=48):
    n = len(buf)
    cc = zstd.ZstdCompressor(level=level)
    t, frame = timeit(lambda: cc.compress(buf))
    print(f"  {name:28s} zstd-{level}  1 frame: {len(frame)/1e6:7.2f} MB ({len(frame)/n*100:5.1f}%)  "
          f"compress {n/t/1e6:7.0f} MB/s", end="")
    dd = zstd.ZstdDecompressor()
    t, _ = timeit(lambda: dd.decompress(frame, max_output_size=n))
    print(f"  decompress {n/t/1e6:7.0f} MB/s (1 thread, 1 frame)")
    # independent frames
    parts = chunks(buf, nchunks)
    frames = [cc.compress(p) for p in parts]
    fsz = sum(len(f) for f in frames)
    psz = len(parts[0])
    print(f"  {'':28s}         {nchunks} frames: {fsz/1e6:7.2f} MB ({fsz/n*100:5.1f}%)", end="")
    for th in (1, 2, 4, 8, 16):
        def run():
            with ThreadPoolExecutor(th) as ex:
                return list(ex.map(lambda f: zstd.ZstdDecompressor().decompress(f, max_output_size=psz), frames))
        t, _ = timeit(run)
        print(f"  dec x{th}: {n/t/1e6:6.0f}", end="")
    print(" MB/s")
    for th in (4, 8):
        def runc():
            with ThreadPoolExecutor(th) as ex:
                return list(ex.map(lambda p: zstd.ZstdCompressor(level=level).compress(p), parts))
        t, _ = timeit(runc, 2)
        print(f"  {'':28s}         compress x{th}: {n/t/1e6:6.0f} MB/s", end="")
    print()
    # zstd's own multithreaded compressor on one buffer
    cm = zstd.ZstdCompressor(level=level, threads=8)
    t, fr8 = timeit(lambda: cm.compress(buf), 2)
    print(f"  {'':28s}         zstd threads=8 (1 frame): compress {n/t/1e6:6.0f} MB/s, {len(fr8)/1e6:.2f} MB")


def bench_lz4(name, buf, nchunks=48):
    n = len(buf)
    t, fr = timeit(lambda: lz4.block.compress(buf.tobytes(), store_size=False))
    t2, _ = timeit(lambda: lz4.block.decompress(fr, uncompressed_size=n))
    print(f"  {name:28s} lz4        : {len(fr)/1e6:7.2f} MB ({len(fr)/n*100:5.1f}%)  compress {n/t/1e6:7.0f} MB/s  "
          f"decompress {n/t2/1e6:7.0f} MB/s (1 thread)")


if __name__ == "__main__":
    path = sys.argv[1]
    h = C.read_head(path)
    s = C.e_sections(h)
    st = np.fromfile(path, dtype=np.uint32, count=C.STATE_BYTES // 4, offset=s["state"])
    raw = st.view(np.uint8)
    pl = np.ascontiguousarray(raw.reshape(-1, 4).T)  # (4, n) low byte first
    hi = np.ascontiguousarray(pl[3])
    split = pl.reshape(-1)
    print(f"cores {os.cpu_count()}; state {len(raw)/1e6:.2f} MB; hi-byte plane {len(hi)/1e6:.2f} MB; zstd {zstd.ZSTD_VERSION}")
    t0 = time.perf_counter()
    t, _ = timeit(lambda: np.ascontiguousarray(raw.reshape(-1, 4).T), 2)
    print(f"numpy plane split (not representative of C): {len(raw)/t/1e6:.0f} MB/s")
    for lvl in (1, 3):
        bench_zstd("raw bytes (no transform)", raw, lvl)
    for lvl in (1, 3):
        bench_zstd("hi-byte plane only", hi, lvl)
    for lvl in (1,):
        bench_zstd("all 4 planes split", split, lvl)
    bench_lz4("raw bytes", raw)
    bench_lz4("hi-byte plane only", hi)
    # disk read of the same file for scale (OS cache state unknown)
    t = time.perf_counter()
    with open(path, "rb") as f:
        n = 0
        while True:
            b = f.read(8 << 20)
            if not b:
                break
            n += len(b)
    dt = time.perf_counter() - t
    print(f"\nplain sequential read of the whole {n/1e6:.0f} MB file (python, 8 MB reads): {dt*1000:.0f} ms = {n/dt/1e6:.0f} MB/s (page-cache state unknown)")
