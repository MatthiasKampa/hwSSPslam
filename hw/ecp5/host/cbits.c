/* cbits — the desc-bit consumer in C ("python bad, use C": the Linux box
 * is a shim; this is its one hot compute). Hardcoded to the trained
 * vision head's geometry (sspax/artifacts/vision_head.npz, headio v2):
 *
 *   gray 120x160 (/255) -> conv3x3 s2 (1->16)  CReLU -> 60x80x32
 *                       -> conv3x3 s2 (32->16) CReLU -> 30x40x32
 *                       -> conv3x3 s1 (32->32) CReLU -> 30x40x64
 *                       -> conv1x1    (64->32) CReLU -> 30x40x64
 *   desc: conv1x1 (64->32) linear -> 30x40x32
 *   pool 2x2 mean, bits = act > 0 -> 15x20x32
 *
 * Arithmetic mirrors sspax.headio.forward (float32 convs on int8*scale
 * dequantized weights, SAME zero pad). Float summation order differs
 * from numpy einsum, so cross-impl bits can differ where a pooled
 * activation sits within ~1e-6 of 0 — the parity gate (cbits.py gate)
 * measures this on real frames; within-impl bits are deterministic.
 *
 * Weights arrive PRE-DEQUANTIZED and PRE-REORDERED from python:
 *   3x3 conv: (ky, kx, cin, cout) contiguous; 1x1 conv: (cin, cout).
 * Single global context; not thread-safe (one worker owns it).
 *
 * build: cc -O3 -march=native -shared -fPIC -o libcbits.so cbits.c
 */
#include <string.h>

#define IH 120
#define IW 160
#define H0 60
#define W0 80
#define C0 16 /* conv cout; CReLU doubles to 32 */
#define H1 30
#define W1 40
#define C1 16
#define C2 32
#define C3 32
#define CD 32

static float w0[3 * 3 * 1 * C0], b0[C0];
static float w1[3 * 3 * (2 * C0) * C1], b1[C1];
static float w2[3 * 3 * (2 * C1) * C2], b2[C2];
static float w3[(2 * C2) * C3], b3[C3];
static float wd[(2 * C3) * CD], bd[CD];

/* padded activation buffers (edges zero once = SAME pad forever) */
static float xin[(IH + 2) * (IW + 2)];
static float a0[(H0 + 2) * (W0 + 2) * (2 * C0)];
static float a1[(H1 + 2) * (W1 + 2) * (2 * C1)];
static float a2[H1 * W1 * (2 * C2)];
static float a3[H1 * W1 * (2 * C3)];
static float dsc[H1 * W1 * CD];

static void conv3x3_crelu(const float *xp, int win_p, int cin,
                          const float *w, const float *b, int cout,
                          int stride, float *outp, int wout_p,
                          int hout, int wout)
{
    for (int oy = 0; oy < hout; oy++) {
        for (int ox = 0; ox < wout; ox++) {
            float acc[C2];
            for (int o = 0; o < cout; o++)
                acc[o] = b[o];
            for (int ky = 0; ky < 3; ky++) {
                const float *xr = xp + ((oy * stride + ky) * win_p
                                        + ox * stride) * cin;
                const float *wr = w + ky * 3 * cin * cout;
                for (int kx = 0; kx < 3; kx++) {
                    const float *xc = xr + kx * cin;
                    const float *wc = wr + kx * cin * cout;
                    for (int c = 0; c < cin; c++) {
                        float xv = xc[c];
                        const float *wo = wc + c * cout;
                        if (xv != 0.0f)
                            for (int o = 0; o < cout; o++)
                                acc[o] += xv * wo[o];
                    }
                }
            }
            float *op = outp + ((oy + 1) * wout_p + (ox + 1)) * (2 * cout);
            for (int o = 0; o < cout; o++) {
                float v = acc[o];
                op[o] = v > 0.0f ? v : 0.0f;
                op[o + cout] = v < 0.0f ? -v : 0.0f;
            }
        }
    }
}

static void conv1x1(const float *x, int n, int cin, const float *w,
                    const float *b, int cout, int crelu, float *out)
{
    for (int i = 0; i < n; i++) {
        const float *xi = x + i * cin;
        float acc[C3];
        for (int o = 0; o < cout; o++)
            acc[o] = b[o];
        for (int c = 0; c < cin; c++) {
            float xv = xi[c];
            const float *wo = w + c * cout;
            if (xv != 0.0f)
                for (int o = 0; o < cout; o++)
                    acc[o] += xv * wo[o];
        }
        if (crelu) {
            float *op = out + i * 2 * cout;
            for (int o = 0; o < cout; o++) {
                float v = acc[o];
                op[o] = v > 0.0f ? v : 0.0f;
                op[o + cout] = v < 0.0f ? -v : 0.0f;
            }
        } else {
            float *op = out + i * cout;
            for (int o = 0; o < cout; o++)
                op[o] = acc[o];
        }
    }
}

int cbits_setup(const float *w0_, const float *b0_, const float *w1_,
                const float *b1_, const float *w2_, const float *b2_,
                const float *w3_, const float *b3_, const float *wd_,
                const float *bd_)
{
    memcpy(w0, w0_, sizeof w0);
    memcpy(b0, b0_, sizeof b0);
    memcpy(w1, w1_, sizeof w1);
    memcpy(b1, b1_, sizeof b1);
    memcpy(w2, w2_, sizeof w2);
    memcpy(b2, b2_, sizeof b2);
    memcpy(w3, w3_, sizeof w3);
    memcpy(b3, b3_, sizeof b3);
    memcpy(wd, wd_, sizeof wd);
    memcpy(bd, bd_, sizeof bd);
    memset(xin, 0, sizeof xin);
    memset(a0, 0, sizeof a0);
    memset(a1, 0, sizeof a1);
    return 0;
}

/* g: 120*160 uint8 row-major; bits: 15*20*32 uint8 out (0/1);
 * desc_or_null: optional 30*40*32 float out (pre-pool activations) */
int cbits_forward(const unsigned char *g, unsigned char *bits,
                  float *desc_or_null)
{
    for (int y = 0; y < IH; y++) {
        float *xr = xin + (y + 1) * (IW + 2) + 1;
        const unsigned char *gr = g + y * IW;
        for (int x = 0; x < IW; x++)
            xr[x] = (float)gr[x] / 255.0f;
    }
    conv3x3_crelu(xin, IW + 2, 1, w0, b0, C0, 2, a0, W0 + 2, H0, W0);
    conv3x3_crelu(a0, W0 + 2, 2 * C0, w1, b1, C1, 2, a1, W1 + 2, H1, W1);
    /* L2: stride 1, output unpadded (a2 feeds only 1x1s) */
    for (int oy = 0; oy < H1; oy++) {
        for (int ox = 0; ox < W1; ox++) {
            float acc[C2];
            for (int o = 0; o < C2; o++)
                acc[o] = b2[o];
            for (int ky = 0; ky < 3; ky++) {
                const float *xr = a1 + ((oy + ky) * (W1 + 2) + ox)
                                  * (2 * C1);
                const float *wr = w2 + ky * 3 * (2 * C1) * C2;
                for (int kx = 0; kx < 3; kx++) {
                    const float *xc = xr + kx * (2 * C1);
                    const float *wc = wr + kx * (2 * C1) * C2;
                    for (int c = 0; c < 2 * C1; c++) {
                        float xv = xc[c];
                        const float *wo = wc + c * C2;
                        if (xv != 0.0f)
                            for (int o = 0; o < C2; o++)
                                acc[o] += xv * wo[o];
                    }
                }
            }
            float *op = a2 + (oy * W1 + ox) * (2 * C2);
            for (int o = 0; o < C2; o++) {
                float v = acc[o];
                op[o] = v > 0.0f ? v : 0.0f;
                op[o + C2] = v < 0.0f ? -v : 0.0f;
            }
        }
    }
    conv1x1(a2, H1 * W1, 2 * C2, w3, b3, C3, 1, a3);
    conv1x1(a3, H1 * W1, 2 * C3, wd, bd, CD, 0, dsc);
    if (desc_or_null)
        memcpy(desc_or_null, dsc, sizeof dsc);
    for (int cy = 0; cy < H1 / 2; cy++) {
        for (int cx = 0; cx < W1 / 2; cx++) {
            const float *p00 = dsc + ((2 * cy) * W1 + 2 * cx) * CD;
            const float *p01 = p00 + CD;
            const float *p10 = p00 + W1 * CD;
            const float *p11 = p10 + CD;
            unsigned char *ob = bits + (cy * (W1 / 2) + cx) * CD;
            for (int c = 0; c < CD; c++)
                ob[c] = (p00[c] + p01[c] + p10[c] + p11[c]) > 0.0f;
        }
    }
    return 0;
}
