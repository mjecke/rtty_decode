#!/usr/bin/env python3
"""
rtty_decode.py - Decode ITA-2 (Baudot) RTTY recordings from a WAV file to text.

Requirements:  pip install numpy scipy

How it works
  1. Reads the WAV file (mono or stereo, 8/16/24/32-bit or float).
  2. Finds the mark/space tone pair automatically (or uses --mark / --space).
  3. Demodulates the FSK with two quadrature (I/Q) tone detectors + low-pass
     filters, giving a normalised soft decision (+1 = mark, -1 = space).
  4. Recovers asynchronous characters: 1 start bit, 5 data bits (LSB first),
     1 / 1.5 / 2 stop bits. Each character is re-synchronised on its start edge,
     so slight baud-rate errors are tolerated.
  5. Converts the 5-bit codes to text using the ITA-2 LTRS/FIGS tables.

If you don't give --baud and/or --polarity, it tries the common options
(45.45, 50, 75, 100 baud; normal and reversed) and keeps the best-looking result.

Examples
  python rtty_decode.py recording.wav
  python rtty_decode.py recording.wav --baud 45.45 --mark 2125 --space 2295
  python rtty_decode.py recording.wav --polarity reverse --usos -o out.txt
"""

import argparse
import re
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d

# --------------------------------------------------------------------------
# ITA-2 tables, indexed by the 5-bit code (first received bit = LSB)
# --------------------------------------------------------------------------
LTRS = [
    "\0", "E", "\n", "A", " ", "S", "I", "U",
    "\r", "D", "R", "J", "N", "F", "C", "K",
    "T", "Z", "L", "W", "H", "Y", "P", "Q",
    "O", "B", "G", "<FIGS>", "M", "X", "V", "<LTRS>",
]
FIGS = [
    "\0", "3", "\n", "-", " ", "'", "8", "7",
    "\r", "$", "4", "[BEL]", ",", "!", ":", "(",
    "5", "+", ")", "2", "#", "6", "0", "1",
    "9", "?", "&", "<FIGS>", ".", "/", "=", "<LTRS>",
]
CODE_FIGS, CODE_LTRS = 27, 31

COMMON_BAUDS = [45.45, 50.0, 75.0, 100.0]
COMMON_SHIFTS = [170, 425, 450, 850]
COMMON_LETTERS = set("ETAOINSRHDLU \n")


def log(*a):
    print(*a, file=sys.stderr)


# --------------------------------------------------------------------------
# Audio input
# --------------------------------------------------------------------------
def read_wav(path):
    fs, data = wavfile.read(path)
    if data.dtype == np.uint8:
        x = (data.astype(np.float64) - 128.0) / 128.0
    elif np.issubdtype(data.dtype, np.integer):
        x = data.astype(np.float64) / float(np.iinfo(data.dtype).max)
    else:
        x = data.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x -= x.mean()
    return fs, x


# --------------------------------------------------------------------------
# Tone detection
# --------------------------------------------------------------------------
def find_tones(x, fs, shifts, fmin=200.0, fmax=6000.0):
    """Find the (low, high) FSK tone pair from the long-term spectrum."""
    nper = int(min(len(x), max(fs // 2, 256)))
    f, p = signal.welch(x, fs, nperseg=nper)
    df = f[1] - f[0]
    ps = uniform_filter1d(p, max(1, int(round(15.0 / df))))

    lo_i = int(fmin / df)
    hi_i = int(min(fmax, 0.95 * fs / 2) / df)
    best = None
    for sh in shifts:
        off = int(round(sh / df))
        idx = np.arange(lo_i, min(hi_i, len(ps) - off - 1))
        if len(idx) == 0:
            continue
        # both tones must be present, so score on the weaker of the two
        sc = np.minimum(ps[idx], ps[idx + off])
        j = int(np.argmax(sc))
        if best is None or sc[j] > best[0]:
            best = (sc[j], sh, int(idx[j]))
    if best is None:
        raise RuntimeError("Could not find an FSK tone pair in this file.")

    _, sh, i = best
    off = int(round(sh / df))
    w = int(round(10.0 / df))

    def refine(k):
        a, b = max(0, k - w), min(len(ps), k + w + 1)
        return f[a + int(np.argmax(ps[a:b]))]

    return refine(i), refine(i + off)


# --------------------------------------------------------------------------
# Demodulation
# --------------------------------------------------------------------------
def tone_envelopes(x, fs, freqs, cutoff):
    """Envelope of each tone via I/Q mixing + zero-phase low-pass."""
    t = np.arange(len(x)) / fs
    sos = signal.butter(4, cutoff, btype="low", fs=fs, output="sos")
    envs = []
    for fr in freqs:
        c = signal.sosfiltfilt(sos, x * np.cos(2 * np.pi * fr * t))
        s = signal.sosfiltfilt(sos, x * np.sin(2 * np.pi * fr * t))
        envs.append(np.hypot(c, s))
    return envs


# --------------------------------------------------------------------------
# Framing (start bit / 5 data bits / stop bit)
# --------------------------------------------------------------------------
def extract_codes(soft, active, spb, keep_errors=False):
    """soft > 0 means mark. Returns (list of 5-bit codes, number of framing errors)."""
    n = len(soft)
    hard = soft > 0
    edges = np.flatnonzero(hard[:-1] & ~hard[1:]) + 1  # mark -> space transitions
    cs = np.concatenate(([0.0], np.cumsum(soft)))
    half = max(1, int(0.2 * spb))

    def avg(c):  # mean soft value in a window around sample c
        c = int(round(c))
        a, b = c - half, c + half + 1
        return (cs[b] - cs[a]) / (b - a)

    codes, errors, pos = [], 0, 0
    ne = len(edges)
    while True:
        k = int(np.searchsorted(edges, pos))
        if k >= ne:
            break
        t0 = int(edges[k])
        if t0 + 7.0 * spb + half + 2 >= n:
            break
        if not active[t0 + int(0.5 * spb)] or avg(t0 + 0.5 * spb) >= 0:
            pos = t0 + 1  # noise glitch, not a real start bit
            continue
        code = 0
        for i in range(5):
            if avg(t0 + (1.5 + i) * spb) > 0:
                code |= 1 << i
        if avg(t0 + 6.5 * spb) > 0:  # valid stop bit
            codes.append(code)
            pos = int(t0 + 6.5 * spb)
        else:
            errors += 1
            if keep_errors:
                codes.append(code)
            pos = t0 + 1
    return codes, errors


# --------------------------------------------------------------------------
# ITA-2 -> text
# --------------------------------------------------------------------------
def codes_to_text(codes, usos=False):
    out, figs = [], False
    for c in codes:
        if c == CODE_FIGS:
            figs = True
            continue
        if c == CODE_LTRS:
            figs = False
            continue
        if c == 0:
            continue
        if c == 4 and usos:  # unshift on space
            figs = False
        out.append((FIGS if figs else LTRS)[c])
    text = "".join(out)
    return re.sub(r"\r+\n|\n\r+|\r", "\n", text)


def score_decode(codes, errors, usos):
    """Heuristic 'does this look like text?' score used for auto-selection."""
    n = len(codes)
    if n < 5:
        return 0.0
    valid_frac = n / (n + errors)
    text = codes_to_text(codes, usos).upper()
    if not text:
        return 0.0
    common = sum(ch in COMMON_LETTERS for ch in text) / len(text)
    return valid_frac * common * (n / (n + 20.0))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Decode ITA-2 RTTY from a WAV file.")
    ap.add_argument("wavfile", help="input .wav recording")
    ap.add_argument("-o", "--output", help="write decoded text to this file")
    ap.add_argument("--baud", type=float, help="baud rate (default: auto among 45.45/50/75/100)")
    ap.add_argument("--shift", type=float, help="expected shift in Hz, narrows tone search (e.g. 170, 425, 850)")
    ap.add_argument("--mark", type=float, help="mark tone frequency in Hz (use with --space)")
    ap.add_argument("--space", type=float, help="space tone frequency in Hz (use with --mark)")
    ap.add_argument("--polarity", choices=["auto", "normal", "reverse"], default="auto",
                    help="normal = mark is the lower tone; reverse = mark is the upper tone")
    ap.add_argument("--usos", action="store_true", help="unshift-on-space (figures shift ends at a space)")
    ap.add_argument("--squelch", type=float, default=0.25,
                    help="ignore signal below this fraction of peak tone level, 0 disables (default 0.25)")
    ap.add_argument("--keep-errors", action="store_true",
                    help="also output characters whose stop bit was invalid")
    args = ap.parse_args()

    if (args.mark is None) != (args.space is None):
        ap.error("--mark and --space must be given together")

    fs, x = read_wav(args.wavfile)
    log(f"Read {len(x) / fs:.1f} s of audio at {fs} Hz")
    if np.max(np.abs(x)) == 0:
        sys.exit("The file is silent.")

    # --- tones --------------------------------------------------------------
    manual = args.mark is not None
    if manual:
        f1, f2 = args.mark, args.space  # f1 is mark in 'normal' polarity
        polarities = ["normal"]
    else:
        shifts = [args.shift] if args.shift else COMMON_SHIFTS
        f1, f2 = find_tones(x, fs, shifts)
        polarities = ["normal", "reverse"] if args.polarity == "auto" else [args.polarity]
        log(f"Detected tones: {f1:.0f} Hz and {f2:.0f} Hz (shift {abs(f2 - f1):.0f} Hz)")

    # --- decimate to a convenient working rate ------------------------------
    factor = max(1, int(fs // max(8000.0, 3.0 * max(f1, f2))))
    if factor > 1:
        x = signal.resample_poly(x, 1, factor)
        fs = fs / factor
    if max(f1, f2) > 0.45 * fs:
        log("Warning: tone frequency is very close to Nyquist; results may be poor.")

    # --- try baud rates / polarities ---------------------------------------
    bauds = [args.baud] if args.baud else COMMON_BAUDS
    best = None
    for baud in bauds:
        cutoff = min(0.75 * baud, 0.4 * abs(f2 - f1))
        e1, e2 = tone_envelopes(x, fs, (f1, f2), cutoff)
        total = e1 + e2
        active = (total > args.squelch * np.percentile(total, 95)) if args.squelch > 0 \
            else np.ones(len(x), dtype=bool)
        base = (e1 - e2) / (total + 1e-12)  # +1 when tone f1 is on
        spb = fs / baud
        for pol in polarities:
            soft = base if pol == "normal" else -base
            codes, errors = extract_codes(soft, active, spb, args.keep_errors)
            sc = score_decode(codes, errors, args.usos)
            log(f"  baud {baud:6.2f}  polarity {pol:7s}  chars {len(codes):6d}  "
                f"framing errors {errors:5d}  score {sc:.3f}")
            if best is None or sc > best[0]:
                best = (sc, baud, pol, codes, errors)

    sc, baud, pol, codes, errors = best
    if sc == 0.0 and not codes:
        log("No RTTY characters could be decoded. Try --baud, --mark/--space, or --squelch 0.")
    else:
        log(f"Using baud {baud}, polarity {pol}: {len(codes)} chars, {errors} framing errors")

    text = codes_to_text(codes, args.usos)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
        log(f"Wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
