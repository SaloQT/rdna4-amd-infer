/* win_compat.h -- the four POSIX pieces amd-infer uses, on Windows.
 *
 * Local port for the rdna4-kernel-lab (Windows 11 + ROCm wheel SDK 10.1,
 * RX 9070 XT / gfx1201). Not upstream. Provides:
 *   dlopen / dlsym / dlclose  -> LoadLibraryA / GetProcAddress / FreeLibrary
 *   mmap / munmap / MAP_*     -> CreateFileMapping + MapViewOfFile
 *   clock_gettime             -> QueryPerformanceCounter
 * Nothing else changes; the engine's logic is untouched.
 */
#ifndef IE_WIN_COMPAT_H
#define IE_WIN_COMPAT_H

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <fcntl.h>
#include <io.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <windows.h>

/* --- posix_memalign ---------------------------------------------------- */
/* ie_alloc asks for 256-byte alignment; the host code reads these buffers
 * with scalar or unaligned accesses and copies them out with hipMemcpy, so
 * malloc's 16-byte alignment is enough, and free() stays valid (MinGW has no
 * posix_memalign; _aligned_malloc would need _aligned_free everywhere). */
static int ie_win_posix_memalign(void **p, size_t align, size_t n) {
  (void)align;
  void *q = malloc(n ? n : 1);
  if (!q) return 1;
  *p = q;
  return 0;
}
#define posix_memalign(p, a, n) ie_win_posix_memalign((p), (a), (n))

/* --- dlfcn ------------------------------------------------------------- */
#ifndef RTLD_NOW
#define RTLD_NOW 2
#endif
static void *ie_win_dlopen(const char *name) { return (void *)LoadLibraryA(name); }
static void *ie_win_dlsym(void *h, const char *sym) { return (void *)GetProcAddress((HMODULE)h, sym); }
static int ie_win_dlclose(void *h) { return FreeLibrary((HMODULE)h) ? 0 : -1; }
#define dlopen(n, f) ie_win_dlopen(n)
#define dlsym(h, s) ie_win_dlsym((h), (s))
#define dlclose(h) ie_win_dlclose(h)

/* --- mmap -------------------------------------------------------------- */
#ifndef PROT_READ
#define PROT_READ 1
#endif
#ifndef MAP_PRIVATE
#define MAP_PRIVATE 2
#endif
#ifndef MAP_FAILED
#define MAP_FAILED ((void *)-1)
#endif
static void *ie_win_mmap(int fd, size_t len) {
  HANDLE fh = (HANDLE)_get_osfhandle(fd);
  if (fh == INVALID_HANDLE_VALUE) return MAP_FAILED;
  /* size 0: map the whole file; the caller passes the file size in len */
  HANDLE mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
  if (!mh) return MAP_FAILED;
  void *p = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, (SIZE_T)len);
  CloseHandle(mh); /* the view keeps its own reference */
  return p ? p : MAP_FAILED;
}
static int ie_win_munmap(void *p) { return UnmapViewOfFile(p) ? 0 : -1; }
#define mmap(addr, len, prot, flags, fd, off) ie_win_mmap((fd), (size_t)(len))
#define munmap(p, l) ie_win_munmap(p)

/* --- clock_gettime ------------------------------------------------------ */
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
/* MinGW's clock_gettime resolves to clock_gettime64, which the import libs do
 * not export. Same semantics from QPC. */
static int ie_win_clock_gettime(struct timespec *ts) {
  LARGE_INTEGER f, c;
  if (!QueryPerformanceFrequency(&f) || !QueryPerformanceFrequency(&f)) return -1;
  QueryPerformanceCounter(&c);
  ts->tv_sec = (time_t)(c.QuadPart / f.QuadPart);
  ts->tv_nsec = (long)((c.QuadPart % f.QuadPart) * 1000000000LL / f.QuadPart);
  return 0;
}
#define clock_gettime(clk, ts) ie_win_clock_gettime(ts)

/* getenv is fine; MSVC-style CRLF output is not used anywhere. */
#endif /* _WIN32 */
#endif /* IE_WIN_COMPAT_H */
