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
