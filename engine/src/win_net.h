/* win_net.h -- the socket pieces ie-serve uses, on Windows (Winsock 2).
 *
 * Local port for the rdna4-kernel-lab, like win_compat.h. Not upstream.
 * Include it instead of the POSIX socket headers; serve.c calls
 * ie_net_init() once before its first socket.
 *   poll                 -> WSAPoll (no POLLRDHUP: a closed peer shows as
 *                           POLLHUP, or POLLIN with a 0-byte peek)
 *   close on a socket    -> closesocket (serve.c closes only sockets)
 *   SO_RCVTIMEO/SNDTIMEO -> a DWORD of milliseconds, not a timeval
 *   SIGPIPE, MSG_NOSIGNAL, MSG_DONTWAIT: not needed (Winsock raises no
 *   signal; the peek runs only after WSAPoll reported data or a close)
 */
#ifndef IE_WIN_NET_H
#define IE_WIN_NET_H

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <io.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif
#ifndef MSG_DONTWAIT
#define MSG_DONTWAIT 0
#endif
#ifndef POLLRDHUP
#define POLLRDHUP 0
#endif

static inline int ie_poll(struct pollfd *p, unsigned long n, int ms) { return WSAPoll(p, n, ms); }
#define poll ie_poll

/* serve.c's fds are ints; a SOCKET value fits on Windows in practice */
static inline int ie_closesocket(int fd) { return closesocket((SOCKET)(uintptr_t)fd); }
#define close ie_closesocket

static inline int ie_send(int fd, const void *p, size_t n, int fl) { return send((SOCKET)(uintptr_t)fd, (const char *)p, (int)n, fl); }
static inline int ie_recv(int fd, void *p, size_t n, int fl) { return recv((SOCKET)(uintptr_t)fd, (char *)p, (int)n, fl); }
#define send ie_send
#define recv ie_recv

static inline int ie_setsockopt(int fd, int level, int name, const void *v, int n) {
  return setsockopt((SOCKET)(uintptr_t)fd, level, name, (const char *)v, n);
}
/* a receive / send timeout of s seconds */
static inline void ie_sock_timeout(int fd, int s) {
  DWORD ms = (DWORD)s * 1000u;
  setsockopt((SOCKET)(uintptr_t)fd, SOL_SOCKET, SO_RCVTIMEO, (const char *)&ms, sizeof ms);
  setsockopt((SOCKET)(uintptr_t)fd, SOL_SOCKET, SO_SNDTIMEO, (const char *)&ms, sizeof ms);
}

static inline void ie_net_init(void) {
  WSADATA w;
  if (WSAStartup(MAKEWORD(2, 2), &w)) {
    fprintf(stderr, "ie-serve: WSAStartup failed\n");
    exit(1);
  }
}

#else /* POSIX */
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#define ie_setsockopt setsockopt
static inline void ie_sock_timeout(int fd, int s) {
  struct timeval tv = {s, 0};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
}
static inline void ie_net_init(void) { signal(SIGPIPE, SIG_IGN); }
#endif

#endif
