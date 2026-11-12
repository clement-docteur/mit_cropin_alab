"""Minimal dBase (.dbf) reader -> pandas DataFrame (no extra packages needed)."""
import struct, pandas as pd
def read_dbf(path):
    with open(path, 'rb') as f:
        hdr = f.read(32)
        nrec, hlen, rlen = struct.unpack('<IHH', hdr[4:12])
        fields = []
        while True:
            d = f.read(32)
            if d[0] == 0x0D: break
            name = d[:11].split(b'\x00')[0].decode()
            fields.append((name, chr(d[11]), d[16], d[17]))
        f.seek(hlen)
        rows = []
        for _ in range(nrec):
            r = f.read(rlen)
            if r[0:1] == b'*': continue
            pos, row = 1, []
            for name, t, ln, dec in fields:
                v = r[pos:pos+ln].decode('latin-1').strip(); pos += ln
                if t in 'NF':
                    try: v = float(v) if v else None
                    except ValueError: v = None
                row.append(v)
            rows.append(row)
    return pd.DataFrame(rows, columns=[f[0] for f in fields])
