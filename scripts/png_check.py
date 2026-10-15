"""PNG に透明な画素(不透明度が 255 未満の画素)があるかを調べる(標準ライブラリのみ)。"""

from __future__ import annotations

import struct
import zlib

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# カラータイプ → 1 画素あたりのチャンネル数
CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def _read_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("PNG の署名がありません")
    chunks = []
    pos = len(PNG_SIGNATURE)
    while pos + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[pos : pos + 8])
        chunks.append((kind, data[pos + 8 : pos + 8 + length]))
        pos += 12 + length
        if kind == b"IEND":
            break
    return chunks


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _unfilter(raw: bytes, height: int, row_bytes: int, bpp: int) -> list[bytearray]:
    rows: list[bytearray] = []
    prev = bytearray(row_bytes)
    pos = 0
    for _ in range(height):
        ftype = raw[pos]
        cur = bytearray(raw[pos + 1 : pos + 1 + row_bytes])
        pos += 1 + row_bytes
        for i in range(row_bytes):
            a = cur[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if ftype == 1:
                cur[i] = (cur[i] + a) & 255
            elif ftype == 2:
                cur[i] = (cur[i] + b) & 255
            elif ftype == 3:
                cur[i] = (cur[i] + ((a + b) >> 1)) & 255
            elif ftype == 4:
                cur[i] = (cur[i] + _paeth(a, b, c)) & 255
            elif ftype != 0:
                raise ValueError(f"未知のフィルタ種別: {ftype}")
        rows.append(cur)
        prev = cur
    return rows


def has_transparent_pixels(data: bytes) -> bool:
    """PNG のバイト列に、不透明でない画素があれば True を返す。
    解釈できない PNG(インターレースなど)は ValueError にする。
    """
    chunks = _read_chunks(data)
    width, height, depth, ctype, _comp, _filt, interlace = struct.unpack(
        ">IIBBBBB", chunks[0][1]
    )
    if interlace:
        raise ValueError("インターレースの PNG は透明な画素の検査に未対応です")
    kinds = {k for k, _ in chunks}
    if ctype in (4, 6):
        pass  # 下で画素を調べる
    elif ctype == 3:
        trns = next((d for k, d in chunks if k == b"tRNS"), None)
        if trns is None:
            return False
        if depth != 8:
            return any(v < 255 for v in trns)  # 保守的に判定(未使用のパレット値も含む)
        palette_alpha = list(trns) + [255] * (256 - len(trns))
        return _indexed_uses_alpha(chunks, width, height, depth, palette_alpha)
    else:
        # 0・2 は tRNS の色キーで透明になる。キーに一致する画素があるかは調べず、あれば透明とみなす。
        return b"tRNS" in kinds

    channels = CHANNELS[ctype]
    bits = depth * channels
    row_bytes = (width * bits + 7) // 8
    bpp = max(1, bits // 8)
    raw = zlib.decompress(b"".join(d for k, d in chunks if k == b"IDAT"))
    rows = _unfilter(raw, height, row_bytes, bpp)
    step = bpp
    offset = bpp - (2 if depth == 16 else 1)  # アルファの先頭バイトの位置
    for row in rows:
        for x in range(offset, row_bytes, step):
            if depth == 16:
                if row[x] != 255 or row[x + 1] != 255:
                    return True
            elif row[x] != 255:
                return True
    return False


def _indexed_uses_alpha(chunks, width, height, depth, palette_alpha) -> bool:
    row_bytes = width
    raw = zlib.decompress(b"".join(d for k, d in chunks if k == b"IDAT"))
    rows = _unfilter(raw, height, row_bytes, 1)
    used = set()
    for row in rows:
        used.update(row)
    return any(palette_alpha[i] < 255 for i in used)
