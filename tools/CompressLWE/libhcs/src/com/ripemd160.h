#ifndef RIPEMD160
#define RIPEMD160

#define RIPEMD160_DIGEST_SIZE 20

#include <stdint.h>

typedef struct {
    uint32_t magic;
    uint32_t h[5];      /* The current hash state */
    uint64_t length;    /* Total number of _bits_ (not bytes) added to the
                           hash.  This includes bits that have been buffered
                           but not not fed through the compression function yet. */
    union {
        uint32_t w[16];
        uint8_t b[64];
    } buf;
    uint8_t bufpos;     /* number of bytes currently in the buffer */
} ripemd160_state;

void ripemd160_init(ripemd160_state *self);
void ripemd160_update(ripemd160_state *self, const unsigned char *p, int length);
void ripemd160_copy(const ripemd160_state *source, ripemd160_state *dest);
int ripemd160_digest(const ripemd160_state *self, unsigned char *out);

#endif
