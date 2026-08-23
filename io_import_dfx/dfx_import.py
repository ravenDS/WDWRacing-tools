# SPDX-FileCopyrightText: 2026 ravenDS
# SPDX-License-Identifier: GPL-3.0-or-later

"""
DFX/VD3 importer for Blender.
Walt Disney World Quest: Magical Racing Tour - Crystal Dynamics (2000)
Based on the Crystal Dynamics / Gex 2 PS1 engine format (DRM).

github.com/ravenDS
"""

import os, struct, tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import bpy, bmesh

@dataclass
class VD3Texture:
    index: int = 0; tag: str = ""; width: int = 0; height: int = 0
    content_width: int = 0; content_height: int = 0; header_offset: int = 0
    data_offset: int = 0; data_length: int = 0; is_first: bool = False
    png_filename: str = ""

@dataclass
class MdlVertex:
    x: float = 0.0; y: float = 0.0; z: float = 0.0

@dataclass
class MdlBone:
    v_first: int = 0xFFFF; v_last: int = 0xFFFF
    local_x: int = 0; local_y: int = 0; local_z: int = 0
    parent_id: int = 0xFFFF
    world_x: float = 0.0; world_y: float = 0.0; world_z: float = 0.0

@dataclass
class MdlPolygon:
    v1: int = 0; v2: int = 0; v3: int = 0; flags_raw: int = 0
    is_textured: bool = False; is_overlay: bool = False
    u0: float = 0.0; v0: float = 0.0; u1: float = 0.0; v1uv: float = 0.0
    u2: float = 0.0; v2uv: float = 0.0; tpage: int = 0; clut: int = 0
    color_r: int = 128; color_g: int = 128; color_b: int = 128

@dataclass
class ExtractedModel:
    name: str = ""; vertices: List[MdlVertex] = field(default_factory=list)
    polygons: List[MdlPolygon] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    bones: List[MdlBone] = field(default_factory=list)

@dataclass
class AnimChannel:
    bone_index: int = 0
    axis: int = 0            # 0=X 1=Y 2=Z
    plane: int = 0           # 0=rotation 1=scale 2=translation
    kind: str = ''           # record encoding: raw / const / nib / seg
    keyframes: List[float] = field(default_factory=list)

@dataclass
class ParsedAnimation:
    name: str = ""
    channel_count: int = 0
    key_count: int = 0
    record_addr: int = 0
    warnings: List[str] = field(default_factory=list)
    channels: List[AnimChannel] = field(default_factory=list)

    @property
    def frame_count(self):
        return self.key_count

@dataclass
class SplineSegment:
    seg_type: int = 0
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    triangles: List[Tuple[int, int, int]] = field(default_factory=list)
    tri_materials: list = field(default_factory=list)  # per-tri: (tpage, u0,v0, u1,v1, u2,v2)
    meta: List[int] = field(default_factory=list)

def ru16(data, offset): return struct.unpack_from('<H', data, offset)[0]
def ri16(data, offset): return struct.unpack_from('<h', data, offset)[0]
def ru32(data, offset): return struct.unpack_from('<I', data, offset)[0]
def ri32(data, offset): return struct.unpack_from('<i', data, offset)[0]

# =========================================================================
#  Bone animation parsing
# =========================================================================
#
# Animation record (16 bytes; the object header's +0x14 field points to a
# table of pointers to these, terminated by a non-pointer value):
#     +0  u16 channel_count
#     +2  u16 key_count
#     +4  u32 -> channel header (bit-mask)
#     +8  u32 -> curve stream
#     +12 u32 -> 8-char name
#
# Channel header:
#     byte 0   plane flags: bit0 = rotation, bit1 = scale, bit2 = translation
#     byte 1.. one bit-plane per set flag, in flag order.  Each plane holds
#              3 bits per bone (bit index = bone*3 + axis, LSB first) and is
#              ceil(anim_bone_count*3/8) bytes.  The animation's bone count
#              may exceed the mesh's (shared kart rigs); the whole header is
#              padded to an even length.
#
# Channels are ordered bone-major; inside each bone: rotation x,y,z,
# translation x,y,z, scale x,y,z (only the flagged ones).  The curve stream
# holds one self-delimiting record per channel, in that order.  With the
# first u16 `w` of a record, mode = w >> 13:
#     000 / 111  raw   : w itself is key 0 (i16), followed by key_count-1 i16
#     001        const : 13-bit signed value in w, held for every key
#     010        nib   : i16 initial value, then key_count-1 packed 4-bit
#                        nibbles (low nibble first, sign-magnitude: bit 3 =
#                        negative, bits 0-2 = magnitude), padded to 2 bytes.
#                        IMA-ADPCM style: the low byte of w is the initial
#                        step index; each nibble adds step/8 + mag*step/4
#                        (step from ANIM_NIB_STEP, index adapted by
#                        ANIM_NIB_INDEX) with the nibble's sign.
#     011        seg   : low 13 bits of w = record length in u16 words
#                        (including w).  i16 initial value, then words with
#                        bits 0-4 = run length in keys and bits 5-15 = signed
#                        delta spread linearly over that run; runs sum to
#                        key_count-1.
#
# Units: rotation 4096 = 360 deg (per-axis Euler, X applied first, then Y,
# then Z i.e. M = Rz*Ry*Rx), scale 1024 = 1.0, translation = absolute bone
# position in parent space (same units as the bone rest offsets).

ANIM_PLANE_ROT, ANIM_PLANE_SCALE, ANIM_PLANE_TRANS = 0, 1, 2
ANIM_PLANE_NAMES = {0: 'rot', 1: 'scale', 2: 'trans'}
ANIM_ROT_UNIT = 4096.0
ANIM_SCALE_UNIT = 1024.0
ANIM_NIB_INDEX = [-1, -2, -1, -1, 1, 2, 4, 7, -1, -2, -1, -1, 1, 2, 4, 7]
ANIM_NIB_STEP = [7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 24, 26, 29, 32, 35, 38,
                 41, 45, 49, 53, 57, 62, 67, 72, 79, 85, 91, 97, 104, 111, 118, 126,
                 107, 115, 124, 133, 142, 152, 162, 173, 184, 196, 208, 221, 234, 248,
                 262, 277, 293, 309, 326, 343, 361, 380, 400, 421, 443, 466, 491, 518,
                 547, 579]


def _sign_extend(v, bits):
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def _anim_record_length(w, keys):
    mode = w >> 13
    if mode in (0, 7):
        return 2 * keys, 'raw'
    if mode == 1:
        return 2, 'const'
    if mode == 2:
        n = 4 + (keys - 1 + 1) // 2
        return n + (n & 1), 'nib'
    if mode == 3:
        return (w & 0x1FFF) * 2, 'seg'
    return None, None


def _anim_decode_record(kind, w, blob, keys):
    if kind == 'raw':
        return [float(ri16(blob, 2 * i)) for i in range(keys)]
    if kind == 'const':
        return [float(_sign_extend(w, 13))] * keys
    if kind == 'nib':
        # IMA-ADPCM style: 4-bit sign-magnitude nibbles scaled by an
        # adaptive step (low byte of the header word = initial step index).
        cur = float(ri16(blob, 2))
        idx = blob[0]
        out = [cur]
        nibs = []
        for b in blob[4:]:
            nibs.append(b & 15)
            nibs.append(b >> 4)
        for n in nibs[:keys - 1]:
            step = ANIM_NIB_STEP[min(max(idx, 0), 63)]
            idx = min(max(idx + ANIM_NIB_INDEX[n], 0), 63)
            mag = n & 7
            delta = step >> 3
            if mag & 4:
                delta += step
            if mag & 2:
                delta += step >> 1
            if mag & 1:
                delta += (step >> 2) + (step & 1)
            cur += -delta if n & 8 else delta
            out.append(cur)
        while len(out) < keys:
            out.append(cur)
        return out
    if kind == 'seg':
        cur = float(ri16(blob, 2))
        out = [cur]
        for i in range(2, len(blob) // 2):
            ww = ru16(blob, 2 * i)
            run = ww & 31
            delta = float(_sign_extend(ww >> 5, 11))
            if run == 0:
                cur += delta
                continue
            for r in range(1, run + 1):
                out.append(cur + delta * r / run)
            cur += delta
        out = out[:keys]
        while len(out) < keys:
            out.append(cur)
        return out
    return [0.0] * keys


def _anim_plane_bits(plane):
    for i in range(len(plane) * 8):
        if (plane[i >> 3] >> (i & 7)) & 1:
            yield i // 3, i % 3


def _anim_channel_list(hdr, mesh_bone_count, channel_count):
    """Ordered list of (plane, bone, axis) described by a channel header."""
    if not hdr:
        return []
    flags = hdr[0]
    present = [k for k in range(3) if flags >> k & 1]
    if not present:
        return []
    body = hdr[1:]
    if len(present) == 1:
        plane_size = len(body)
    else:
        plane_size = (mesh_bone_count * 3 + 7) // 8
        if 1 + plane_size * len(present) > len(hdr):
            plane_size = len(body) // len(present)
    sets = {}
    pos = 0
    for k in present:
        sets[k] = set(_anim_plane_bits(body[pos:pos + plane_size]))
        pos += plane_size
    max_bones = plane_size * 8 // 3 + 1
    out = []
    bone_order = [k for k in (ANIM_PLANE_ROT, ANIM_PLANE_TRANS, ANIM_PLANE_SCALE) if k in present]
    for b in range(max_bones):
        for k in bone_order:
            for ax in range(3):
                if (b, ax) in sets[k]:
                    out.append((k, b, ax))
        if len(out) >= channel_count:
            break
    return out[:channel_count]


def _unwrap_rotation(values):
    """Make a 4096-unit rotation channel continuous (raw channels wrap)."""
    if not values:
        return values
    out = [values[0]]
    for v in values[1:]:
        prev = out[-1]
        while v - prev > ANIM_ROT_UNIT / 2:
            v -= ANIM_ROT_UNIT
        while v - prev < -ANIM_ROT_UNIT / 2:
            v += ANIM_ROT_UNIT
        out.append(v)
    return out


def read_anim_pointers(data, obj_addr, limit=64):
    val14 = ru32(data, obj_addr + 0x14)
    ptrs = []
    if not (0 < val14 < len(data)):
        return ptrs
    for j in range(limit):
        off = val14 + j * 4
        if off + 4 > len(data):
            break
        v = ru32(data, off)
        if 0x20 < v < len(data) - 16:
            ptrs.append(v)
        else:
            break
    return ptrs


def parse_animations(data, obj_addr, mesh_bone_count):
    """Parse every bone animation of an object. Returns a list of ParsedAnimation."""
    results = []
    for ai, p in enumerate(read_anim_pointers(data, obj_addr)):
        nch = ru16(data, p)
        keys = ru16(data, p + 2)
        p_hdr = ru32(data, p + 4)
        p_crv = ru32(data, p + 8)
        p_name = ru32(data, p + 12)
        if nch == 0 or keys == 0 or keys > 4096:
            continue
        if not (0 < p_hdr < p_crv < len(data)):
            continue
        anim = ParsedAnimation()
        anim.record_addr = p
        anim.channel_count = nch
        anim.key_count = keys
        anim.name = f"anim_{ai}"
        if 0 < p_name < len(data) - 8:
            raw = data[p_name:p_name + 8]
            e = raw.find(0)
            try:
                nm = (raw[:e] if e >= 0 else raw).decode('ascii').strip('_ ')
                if nm:
                    anim.name = nm
            except Exception:
                pass
        hdr = data[p_hdr:p_crv]
        chlist = _anim_channel_list(hdr, mesh_bone_count, nch)
        if len(chlist) != nch:
            anim.warnings.append(
                f"mask yields {len(chlist)} channels, header says {nch}")
        pos = p_crv
        for ci in range(nch):
            if pos + 2 > len(data):
                anim.warnings.append("curve stream truncated")
                break
            w = ru16(data, pos)
            ln, kind = _anim_record_length(w, keys)
            if ln is None or pos + ln > len(data):
                anim.warnings.append(f"bad record at 0x{pos:X}")
                break
            vals = _anim_decode_record(kind, w, data[pos:pos + ln], keys)
            pos += ln
            if ci < len(chlist):
                k, b, ax = chlist[ci]
                if k == ANIM_PLANE_ROT:
                    vals = _unwrap_rotation(vals)
                anim.channels.append(AnimChannel(
                    bone_index=b, axis=ax, plane=k, kind=kind, keyframes=vals))
        results.append(anim)
    return results



def _is_tag_byte(b: int) -> bool:
    return (65 <= b <= 90) or (97 <= b <= 122) or b == 95 or (48 <= b <= 57)


def _read_tag(data: bytes, offset: int, max_len: int = 12) -> Optional[str]:
    if offset + max_len > len(data):
        return None
    chars = []
    for i in range(max_len):
        b = data[offset + i]
        if b == 0:
            break
        if b < 32 or b > 126:
            return None
        chars.append(chr(b))
    if len(chars) == 0:
        return None
    return ''.join(chars)


def _has_letter(tag: str) -> bool:
    has_l = False
    for c in tag:
        if c.isalpha():
            has_l = True
        elif not (c.isalnum() or c == '_'):
            return False
    return has_l


def scan_vd3(data: bytes) -> List[VD3Texture]:
    """Scan a VD3 file and return a list of texture entries."""
    results = []
    pos = 0
    while pos <= len(data) - 26:
        if not _is_tag_byte(data[pos + 12]):
            pos += 2
            continue

        tag = _read_tag(data, pos + 12, 12)
        if tag is None or len(tag) < 3 or not _has_letter(tag):
            pos += 2
            continue

        w = ru16(data, pos + 4)
        h = ru16(data, pos + 6)
        cw = ru16(data, pos + 8)
        ch = ru16(data, pos + 10)

        # Normal case: w and h are valid
        valid_wh = (1 <= w <= 512 and 1 <= h <= 512)
        valid_cwch = (1 <= cw <= 512 and 1 <= ch <= 512)

        if not valid_cwch:
            pos += 2
            continue

        if valid_wh:
            pix_bytes = w * h * 2
        elif valid_cwch:
            # Fallback: use cw*ch for special textures (e.g. T4VSCRN)
            # where w/h are encoded differently but cw/ch are valid
            w = cw
            h = ch
            pix_bytes = cw * ch * 2
        else:
            pos += 2
            continue

        data_start = pos + 24
        if data_start + pix_bytes > len(data):
            pos += 2
            continue

        tex = VD3Texture()
        tex.index = len(results)
        tex.tag = tag
        tex.width = w
        tex.height = h
        tex.content_width = cw
        tex.content_height = ch
        tex.header_offset = pos
        tex.data_offset = data_start
        tex.data_length = pix_bytes
        tex.is_first = (len(results) == 0)

        safe = ''.join(c if (c.isalnum() or c == '_') else '_' for c in tag)
        if not safe:
            safe = "unnamed"
        tex.png_filename = f"tex_{tex.index:03d}_{safe}.png"

        results.append(tex)
        pos = data_start + pix_bytes

    return results


def decode_vd3_texture_rgba(data: bytes, tex: VD3Texture) -> bytes:
    """Decode a VD3 texture to raw RGBA bytes (4 bytes per pixel).
    Each texture has 2 padding u16 (0x0000) at the start of the data area.
    Real pixels start at data_offset+4, with the last 2 pixels overflowing
    4 bytes past the declared data boundary into the next texture's header."""
    w, h = tex.width, tex.height
    total_pixels = w * h

    # Skip 2 padding pixels (4 bytes), read W*H pixels from data+4
    pixels = [0] * total_pixels
    src_off = tex.data_offset + 4
    for i in range(total_pixels):
        off = src_off + i * 2
        if off + 2 <= len(data):
            pixels[i] = ru16(data, off)

    # Convert RGB555 to RGBA8888
    rgba = bytearray(total_pixels * 4)
    for i in range(total_pixels):
        raw = pixels[i]
        if raw == 0:
            rgba[i*4:i*4+4] = b'\x00\x00\x00\x00'
        else:
            r5 = raw & 0x1F
            g5 = (raw >> 5) & 0x1F
            b5 = (raw >> 10) & 0x1F
            rgba[i*4]     = (r5 << 3) | (r5 >> 2)
            rgba[i*4 + 1] = (g5 << 3) | (g5 >> 2)
            rgba[i*4 + 2] = (b5 << 3) | (b5 >> 2)
            rgba[i*4 + 3] = 255

    return bytes(rgba)


def _deinterlace_rgba(rgba, w, h):
    """PS1 stippled semi-transparency: some animated-effect base textures
    (e.g. x4foam01) store every other row fully transparent so the hardware
    dithers them.  Detect the pattern and fill the empty rows from their
    neighbours so the texture looks like it does on screen."""
    if h < 4:
        return rgba
    def row_alpha(y):
        base = y * w * 4
        return sum(rgba[base + 3 + x * 4] for x in range(w))
    empty_even = all(row_alpha(y) == 0 for y in range(0, h, 2))
    empty_odd = all(row_alpha(y) == 0 for y in range(1, h, 2))
    if empty_even == empty_odd:
        return rgba
    out = bytearray(rgba)
    start = 0 if empty_even else 1
    stride = w * 4
    for y in range(start, h, 2):
        src = y + 1 if y + 1 < h else y - 1
        out[y * stride:(y + 1) * stride] = rgba[src * stride:(src + 1) * stride]
    return bytes(out)


def save_texture_png(data: bytes, tex: VD3Texture, out_path: str):
    """Save a VD3 texture as a PNG file using pure Python (minimal dependencies)."""
    import zlib

    w, h = tex.width, tex.height
    rgba = decode_vd3_texture_rgba(data, tex)
    rgba = _deinterlace_rgba(rgba, w, h)

    # Build PNG manually (no PIL dependency needed in Blender)
    def _make_chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
        c = chunk_type + chunk_data
        crc = struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack('>I', len(chunk_data)) + c + crc

    # IHDR
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)  # 8-bit RGBA

    # IDAT — raw image data with filter byte 0 per row
    raw_rows = bytearray()
    for y in range(h):
        raw_rows.append(0)  # filter: None
        row_start = y * w * 4
        raw_rows.extend(rgba[row_start:row_start + w * 4])

    compressed = zlib.compress(bytes(raw_rows), 9)

    png = b'\x89PNG\r\n\x1a\n'
    png += _make_chunk(b'IHDR', ihdr)
    png += _make_chunk(b'IDAT', compressed)
    png += _make_chunk(b'IEND', b'')

    with open(out_path, 'wb') as f:
        f.write(png)


# =====================================================================
#  DFX parsing
# =====================================================================


def _read_vertices(data: bytes, start: int, count: int) -> List[MdlVertex]:
    verts = []
    for i in range(count):
        off = start + i * 8
        if off + 6 > len(data):
            break
        x = ri16(data, off)
        y = ri16(data, off + 2)
        z = ri16(data, off + 4)
        verts.append(MdlVertex(float(x), float(y), float(z)))
    return verts


def _read_bones(data: bytes, start: int, count: int) -> List[MdlBone]:
    bones = []
    for i in range(count):
        off = start + i * 24
        if off + 20 > len(data):
            break
        b = MdlBone()
        b.v_first = ru16(data, off + 8)
        b.v_last = ru16(data, off + 10)
        b.local_x = ri16(data, off + 12)
        b.local_y = ri16(data, off + 14)
        b.local_z = ri16(data, off + 16)
        b.parent_id = ru16(data, off + 18)
        bones.append(b)

    # Accumulate world positions
    for i in range(len(bones)):
        if bones[i].v_first == 0xFFFF or bones[i].v_last == 0xFFFF:
            continue
        aid = i
        safety = 0
        while 0 <= aid < len(bones) and safety < 256:
            bones[i].world_x += float(bones[aid].local_x)
            bones[i].world_y += float(bones[aid].local_y)
            bones[i].world_z += float(bones[aid].local_z)
            if bones[aid].parent_id == aid:
                break
            if bones[aid].parent_id == 0xFFFF:
                break
            aid = bones[aid].parent_id
            safety += 1

    return bones


def _apply_armature(verts: List[MdlVertex], bones: List[MdlBone]):
    for b in bones:
        if b.v_first == 0xFFFF or b.v_last == 0xFFFF:
            continue
        last_v = min(b.v_last, len(verts) - 1)
        for v in range(b.v_first, last_v + 1):
            verts[v].x += b.world_x
            verts[v].y += b.world_y
            verts[v].z += b.world_z


def _read_material(data: bytes, mat_addr: int, poly: MdlPolygon):
    if mat_addr < 0 or mat_addr + 10 > len(data):
        poly.is_textured = False
        poly.color_r = 128; poly.color_g = 128; poly.color_b = 128
        return

    poly.is_textured = True
    poly.u0 = data[mat_addr] / 255.0
    poly.v0 = data[mat_addr + 1] / 255.0
    poly.clut = ru16(data, mat_addr + 2)
    poly.u1 = data[mat_addr + 4] / 255.0
    poly.v1uv = data[mat_addr + 5] / 255.0
    poly.tpage = ru16(data, mat_addr + 6)
    poly.u2 = data[mat_addr + 8] / 255.0
    poly.v2uv = data[mat_addr + 9] / 255.0


def _resolve_tpage(raw_tpage: int, tex_count: int) -> int:
    if raw_tpage < tex_count:
        return raw_tpage
    masked = raw_tpage & 0x1FF
    if masked < tex_count:
        return masked
    return -1


def extract_level_geometry(data: bytes) -> tuple:
    """Returns (main_model, overlay_list). overlay_list is a list of dicts per surface."""
    lp = ru32(data, 0)
    vc = ru32(data, lp + 0x18)
    pc = ru32(data, lp + 0x20)
    vs = ru32(data, lp + 0x30)
    ps = ru32(data, lp + 0x38)
    ms = ru32(data, lp + 0x44)

    verts = _read_vertices(data, vs, vc)
    polys = []

    # Read overlay effect type table from lp+0x48
    effect_types = []
    interlaced_to_clean = {}  # base_tex → first_frame_tex substitution
    p48 = ru32(data, lp + 0x48) if lp + 0x4C <= len(data) else 0
    if p48 > 0 and p48 < len(data) - 4:
        effect_count = ru32(data, p48)
        if 0 < effect_count < 100:
            for ei in range(effect_count):
                sub_ptr = ru32(data, p48 + 4 + ei * 4)
                if 0 < sub_ptr < len(data) - 0x20:
                    et = {
                        'index': ei, 'ptr': sub_ptr,
                        'base_tex': ru32(data, sub_ptr),
                        'width': ru32(data, sub_ptr + 4),
                        'height': ru32(data, sub_ptr + 8),
                        'param_0C': ru32(data, sub_ptr + 0x0C),
                        'param_10': ru32(data, sub_ptr + 0x10),
                        'param_14': ru32(data, sub_ptr + 0x14),
                        'anim_type': ru32(data, sub_ptr + 0x18),
                        'frame_count': ru32(data, sub_ptr + 0x1C),
                    }
                    frames = []
                    fc = et['frame_count'] if et['frame_count'] < 64 else 0
                    for fi in range(fc):
                        foff = sub_ptr + 0x20 + fi * 8
                        if foff + 8 > len(data): break
                        frames.append({'tex': ru32(data, foff), 'param': ru32(data, foff + 4)})
                    et['frames'] = frames
                    effect_types.append(et)
                    # Map interlaced base texture → clean first frame
                    if frames and et['base_tex'] != frames[0]['tex']:
                        interlaced_to_clean[et['base_tex']] = frames[0]['tex']

    # Read overlay table from lp+0x28/+0x2C
    overlay_count = ru32(data, lp + 0x28)
    overlay_ptr = ru32(data, lp + 0x2C)
    overlay_table_raw = []

    # Build surface_index → effect_type mapping from overlay table
    surface_effect_map = {}  # surface_index → effect_type_index
    if 0 < overlay_count < 1000 and 0 < overlay_ptr < len(data):
        # Align to even address for u32 reads
        tbl = overlay_ptr + (1 if overlay_ptr % 2 == 1 else 0)
        # Auto-detect best start offset (0 or +2) by checking which gives
        # more entries with valid effect types
        best_tbl = tbl
        if effect_types:
            ec = len(effect_types)
            for try_off in (0, 2):
                hits = sum(1 for si in range(min(5, overlay_count))
                           if tbl + try_off + si * 8 + 4 <= len(data)
                           and ru32(data, tbl + try_off + si * 8) < ec)
                if try_off == 0:
                    best_hits = hits
                elif hits > best_hits:
                    best_tbl = tbl + 2
        tbl = best_tbl

        for si in range(overlay_count):
            addr = tbl + si * 8
            if addr + 8 > len(data): break
            v0 = ru32(data, addr); v1 = ru32(data, addr + 4)
            entry = {'v0': v0, 'v1': v1}
            # Detect effect type: first u32 if < effect_count, else try second
            et = 0  # default
            if effect_types:
                ec = len(effect_types)
                if v0 < ec:
                    et = v0
                elif v1 < ec:
                    et = v1
            surface_effect_map[si] = et
            # Check for VFX pointer in either field
            for vcheck in (v0, v1):
                if 0x1000 < vcheck < len(data) - 48:
                    entry['vfx_ptr'] = vcheck
                    entry['vfx_raw'] = ' '.join(f'{data[vcheck+j]:02X}' for j in range(min(48, len(data)-vcheck)))
                    break
            overlay_table_raw.append(entry)

    # Collect overlay polygons grouped by surface index
    overlay_surfaces = {}  # surface_index → dict
    for i in range(pc):
        p_off = ps + i * 12
        if p_off + 12 > len(data):
            break

        p = MdlPolygon()
        p.v1 = ru16(data, p_off)
        p.v2 = ru16(data, p_off + 2)
        p.v3 = ru16(data, p_off + 4)
        p.flags_raw = data[p_off + 7]
        byte6 = data[p_off + 6]
        mat_off = ru16(data, p_off + 10)

        if byte6 == 0x05:
            surf_idx = mat_off
            val8 = ru16(data, p_off + 8)
            p.is_overlay = True
            # Look up effect type for this surface — use clean frame texture
            et_idx = surface_effect_map.get(surf_idx, 0)
            if effect_types and 0 <= et_idx < len(effect_types):
                et = effect_types[et_idx]
                p.is_textured = True
                # Use first frame texture (clean) instead of interlaced base
                if et['frames']:
                    p.tpage = et['frames'][0]['tex']
                else:
                    p.tpage = interlaced_to_clean.get(et['base_tex'], et['base_tex'])
                p.u0 = 0; p.v0 = 0; p.u1 = 0; p.v1uv = 1.0
                p.u2 = 1.0; p.v2uv = 0
            else:
                p.is_textured = False
                p.color_r = 80; p.color_g = 80; p.color_b = 80

            if surf_idx not in overlay_surfaces:
                overlay_surfaces[surf_idx] = {
                    'surface_index': surf_idx,
                    'polys': [],
                    'val8_list': [],
                    'flags_list': [],
                    'poly_raw': [],
                }
            surf = overlay_surfaces[surf_idx]
            surf['polys'].append(p)
            surf['val8_list'].append(val8)
            surf['flags_list'].append(p.flags_raw)
            surf['poly_raw'].append(' '.join(f'{data[p_off+j]:02X}' for j in range(12)))
        elif mat_off != 0xFFFF:
            _read_material(data, ms + mat_off, p)
            # Substitute interlaced effect base textures with clean frame versions
            if p.is_textured and interlaced_to_clean:
                resolved = p.tpage if p.tpage < 0x10000 else (p.tpage & 0x1FF)
                if resolved in interlaced_to_clean:
                    p.tpage = interlaced_to_clean[resolved]
            polys.append(p)
        else:
            p.is_textured = False
            p.color_r = 128; p.color_g = 128; p.color_b = 128
            polys.append(p)

    mdl = ExtractedModel()
    mdl.name = "level_geometry"
    mdl.vertices = verts
    mdl.polygons = polys

    # Build overlay list with all metadata
    overlay_list = []
    for surf_idx in sorted(overlay_surfaces.keys()):
        surf = overlay_surfaces[surf_idx]
        entry = {
            'surface_index': surf_idx,
            'polys': surf['polys'],
            'vertices': verts,
            'val8_list': surf['val8_list'],
            'flags_list': surf['flags_list'],
            'poly_raw': surf['poly_raw'],
            'effect_types': effect_types,
            'effect_type_index': surface_effect_map.get(surf_idx, 0),
            'overlay_table_entry': overlay_table_raw[surf_idx] if surf_idx < len(overlay_table_raw) else None,
            'overlay_count': overlay_count,
        }
        overlay_list.append(entry)

    return mdl, overlay_list


def extract_object_models(data: bytes) -> List[tuple]:
    """Extract all object models from the DFX data section.
    Returns list of (ExtractedModel, obj_addr) tuples."""
    results = []

    models_start = ru32(data, 0x3C)
    pos = models_start
    model_idx = 0

    while True:
        if pos + 4 > len(data):
            break
        obj_addr = ru32(data, pos)
        if obj_addr == models_start:
            break
        pos += 4
        model_idx += 1
        if model_idx > 8000:
            break

        try:
            models = _extract_single_object(data, obj_addr, model_idx)
            if models:
                for mdl in models:
                    results.append((mdl, obj_addr))
        except Exception:
            continue

    return results


def _clean_obj_name(raw_name: str) -> str:
    """Strip trailing underscores/padding from 8-char Crystal Dynamics names."""
    name = raw_name.rstrip('_').strip()
    return name if name else "unnamed"


def _read_obj_name(data: bytes, addr: int) -> str:
    """Read a Crystal Dynamics 8-char padded name from a pointer."""
    if addr <= 0 or addr >= len(data) - 12:
        return ""
    raw = data[addr:addr + 12]
    end = raw.find(0)
    if end > 0:
        try:
            return raw[:end].decode('ascii').strip()
        except Exception:
            return ""
    elif end < 0:
        try:
            return raw.decode('ascii').strip()
        except Exception:
            return ""
    return ""


def _extract_single_object(data: bytes, obj_addr: int,
                            table_index: int = 0) -> List[ExtractedModel]:
    results = []

    if obj_addr + 0x30 > len(data):
        return results

    sub_count = ru16(data, obj_addr + 0x0C)
    sub_start = ru32(data, obj_addr + 0x10)

    # --- Read both name pointers ---
    name1_addr = ru32(data, obj_addr + 0x24)
    name2_addr = ru32(data, obj_addr + 0x28)

    raw_name1 = _read_obj_name(data, name1_addr)
    raw_name2 = _read_obj_name(data, name2_addr)

    clean1 = _clean_obj_name(raw_name1)
    clean2 = _clean_obj_name(raw_name2)

    # Use name2 (descriptive) as primary; fall back to name1 (type)
    if clean2 and clean2 != "unnamed" and clean2 != clean1:
        display_name = clean2
    elif clean1 and clean1 != "unnamed":
        display_name = clean1
    else:
        display_name = "unnamed"

    # --- Read object metadata ---
    obj_flags = ru32(data, obj_addr + 0x00)
    obj_param1 = ru32(data, obj_addr + 0x04)
    val14 = ru32(data, obj_addr + 0x14)
    val18 = ru32(data, obj_addr + 0x18)
    val1C = ru32(data, obj_addr + 0x1C)
    val20 = ru32(data, obj_addr + 0x20)
    val2C = ru32(data, obj_addr + 0x2C)

    # Draw distance / bounding from +0x18, +0x1C (two i16 pairs)
    draw_near = ri16(data, obj_addr + 0x18)
    draw_far = ri16(data, obj_addr + 0x1A)
    bound_near = ri16(data, obj_addr + 0x1C)
    bound_far = ri16(data, obj_addr + 0x1E)

    has_anim = (0 < val14 < len(data))
    has_instance = (0 < val2C < len(data)) and (ru16(data, val2C) == 0xEFBE)
    instance_count = ru16(data, val2C + 2) if has_instance else 0

    metadata = {
        'type_name': clean1,
        'desc_name': clean2,
        'obj_flags': f"0x{obj_flags:08X}",
        'obj_addr': f"0x{obj_addr:X}",
        'table_index': table_index,
        'draw_range': f"({draw_near}, {draw_far})",
        'bound_range': f"({bound_near}, {bound_far})",
        'has_animation': has_anim,
        'has_instance_data': has_instance,
        'instance_state_count': instance_count,
    }

    # Count animation pointers and read animation names if present
    if has_anim:
        anim_count = 0
        anim_ptrs = []
        for j in range(0, 40, 4):
            v = ru32(data, val14 + j)
            if 0x1000 < v < len(data):
                anim_ptrs.append(v)
                anim_count += 1
            else:
                break
        metadata['anim_pointer_count'] = anim_count

        # Parse animation names and frame counts
        anim_list = []
        for aptr in anim_ptrs:
            if aptr + 16 > len(data):
                continue
            frame_count = ru16(data, aptr)
            w1 = ru16(data, aptr + 2)
            name_ptr = ru32(data, aptr + 12)
            anim_name = "?"
            if 0x1000 < name_ptr < len(data) - 12:
                raw = data[name_ptr:name_ptr + 12]
                end = raw.find(0)
                if end > 0:
                    try:
                        anim_name = raw[:end].decode('ascii')
                    except Exception:
                        pass
            anim_list.append({
                'name': anim_name,
                'frames': frame_count,
                'w1': w1,
                'ptr': aptr,
            })
        metadata['animations'] = str([(a['name'], a['frames']) for a in anim_list])
        metadata['anim_names'] = ", ".join(
            f"{a['name']}({a['frames']}f)" for a in anim_list)

    for si in range(sub_count):
        try:
            if sub_start + si * 4 + 4 > len(data):
                continue
            mp = ru32(data, sub_start + si * 4)
            if mp + 0x1C > len(data):
                continue

            vc = ru16(data, mp)
            pc = ru16(data, mp + 4)
            bc = ru16(data, mp + 6)
            v_start = ru32(data, mp + 8)
            p_start = ru32(data, mp + 0x14)
            b_start = ru32(data, mp + 0x18)

            if vc == 0 or pc == 0 or v_start == 0 or p_start == 0:
                continue

            verts = _read_vertices(data, v_start, vc)

            # Read and apply bone armature if present
            model_bones = []
            if bc > 0 and b_start > 0:
                model_bones = _read_bones(data, b_start, bc)
                _apply_armature(verts, model_bones)

            # Read polygons (object style: material address at +8)
            polys = []
            for pi in range(pc):
                p_off = p_start + pi * 12
                if p_off + 12 > len(data):
                    break

                p = MdlPolygon()
                p.v1 = ru16(data, p_off)
                p.v2 = ru16(data, p_off + 2)
                p.v3 = ru16(data, p_off + 4)
                p.flags_raw = data[p_off + 7]

                if (p.flags_raw & 0x02) == 0x02:
                    mat_addr = ru32(data, p_off + 8)
                    _read_material(data, mat_addr, p)
                else:
                    p.is_textured = False
                    p.color_r = data[p_off + 8]
                    p.color_g = data[p_off + 9]
                    p.color_b = data[p_off + 10]

                polys.append(p)

            suffix = f"_sub{si+1}" if sub_count > 1 else ""
            mdl = ExtractedModel()
            mdl.name = f"{table_index:02d}_{display_name}{suffix}"
            mdl.vertices = verts
            mdl.polygons = polys
            mdl.metadata = metadata.copy()
            mdl.bones = model_bones
            results.append(mdl)

        except Exception:
            continue

    return results


# =================================
#  Blender mesh / material creation
# =================================

def _safe_name(name: str) -> str:
    result = ''.join(c if (c.isalnum() or c == '_' or c == '-') else '_' for c in name)
    return result if result else "unnamed"


def _get_or_create_material_textured(
    tpage: int,
    textures: List[VD3Texture],
    tex_images: dict,
    mat_cache: dict,
) -> Optional[bpy.types.Material]:
    """Get or create a Blender material for a textured polygon."""

    resolved = _resolve_tpage(tpage, len(textures))
    if resolved < 0:
        return None

    cache_key = f"tex_{resolved:04d}"
    if cache_key in mat_cache:
        return mat_cache[cache_key]

    tex = textures[resolved]
    mat = bpy.data.materials.new(name=f"mat_{cache_key}_{_safe_name(tex.tag)}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear defaults
    for n in nodes:
        nodes.remove(n)

    # Principled BSDF
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    # Handle Blender 3.x ("Specular") vs 4.x ("Specular IOR Level")
    for spec_name in ('Specular IOR Level', 'Specular'):
        if spec_name in bsdf.inputs:
            bsdf.inputs[spec_name].default_value = 0.0
            break
    bsdf.inputs['Roughness'].default_value = 1.0

    # Output
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    # Texture image
    if resolved in tex_images:
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.location = (-400, 0)
        tex_node.image = tex_images[resolved]
        tex_node.interpolation = 'Closest'  # Pixel art / retro style
        links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(tex_node.outputs['Alpha'], bsdf.inputs['Alpha'])
        # Set alpha clipping — Blender 3.x uses blend_method, 4.x uses different APIs
        try:
            mat.blend_method = 'CLIP'  # Blender 3.x EEVEE
        except Exception:
            pass
        try:
            mat.surface_render_method = 'DITHERED'  # Blender 4.x EEVEE
        except Exception:
            pass

    # Marker textures the engine never draws: N4*DRAW* ("no draw"
    # collision-only surfaces) and A4GUIDE (AI guide surface) are
    # solid-colour placeholders in the VD3.  Hide them like the game does.
    tag = (tex.tag or "").upper()
    if (tag.startswith("N4") and "DRAW" in tag) or tag == "A4GUIDE":
        try:
            bsdf.inputs['Alpha'].default_value = 0.0
            for lk in list(links):
                if lk.to_socket.name == 'Alpha':
                    links.remove(lk)
            mat.blend_method = 'BLEND'
        except Exception:
            pass
        try:
            mat.surface_render_method = 'BLENDED'
        except Exception:
            pass
        mat["dfx_nodraw"] = True

    mat_cache[cache_key] = mat
    return mat


def _get_or_create_material_colored(
    r: int, g: int, b: int,
    mat_cache: dict,
) -> bpy.types.Material:
    """Get or create a Blender material for a flat-colored polygon."""

    key = f"rgb_{r:02X}{g:02X}{b:02X}"
    if key in mat_cache:
        return mat_cache[key]

    mat = bpy.data.materials.new(name=f"mat_{key}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for n in nodes:
        nodes.remove(n)

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    bsdf.inputs['Base Color'].default_value = (r / 255.0, g / 255.0, b / 255.0, 1.0)
    for spec_name in ('Specular IOR Level', 'Specular'):
        if spec_name in bsdf.inputs:
            bsdf.inputs[spec_name].default_value = 0.0
            break
    bsdf.inputs['Roughness'].default_value = 1.0

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    mat_cache[key] = mat
    return mat


def create_blender_mesh(
    model: ExtractedModel,
    textures: List[VD3Texture],
    tex_images: dict,
    mat_cache: dict,
    scale: float,
    collection: bpy.types.Collection,
) -> Optional[bpy.types.Object]:
    """Create a Blender mesh object from an ExtractedModel."""

    if not model.vertices or not model.polygons:
        return None

    name = _safe_name(model.name)
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)

    # --- Gather materials needed by this model ---
    # Map: (is_textured, key) → material_index
    mat_index_map = {}
    mat_list = []

    for poly in model.polygons:
        if poly.is_textured:
            resolved = _resolve_tpage(poly.tpage, len(textures))
            key = ('tex', resolved if resolved >= 0 else -1)
        else:
            key = ('rgb', poly.color_r, poly.color_g, poly.color_b)

        if key not in mat_index_map:
            if key[0] == 'tex' and key[1] >= 0:
                mat = _get_or_create_material_textured(
                    poly.tpage, textures, tex_images, mat_cache)
            elif key[0] == 'rgb':
                mat = _get_or_create_material_colored(
                    key[1], key[2], key[3], mat_cache)
            else:
                mat = _get_or_create_material_colored(128, 128, 128, mat_cache)

            if mat is not None:
                mat_index_map[key] = len(mat_list)
                mat_list.append(mat)
            else:
                mat_index_map[key] = 0

    for mat in mat_list:
        mesh.materials.append(mat)

    # --- Build geometry using BMesh for reliable face/loop ordering ---
    verts_co = [(v.x * scale, v.y * scale, v.z * scale)
                for v in model.vertices]

    bm = bmesh.new()

    # Add all vertices
    bm_verts = [bm.verts.new(co) for co in verts_co]
    bm.verts.ensure_lookup_table()

    # Create UV layer
    uv_layer = bm.loops.layers.uv.new("UVMap")

    # Filter and create faces
    max_vi = len(model.vertices) - 1

    for poly in model.polygons:
        if not (0 <= poly.v1 <= max_vi and
                0 <= poly.v2 <= max_vi and
                0 <= poly.v3 <= max_vi and
                poly.v1 != poly.v2 and poly.v2 != poly.v3 and poly.v1 != poly.v3):
            continue

        try:
            face = bm.faces.new((bm_verts[poly.v1],
                                  bm_verts[poly.v2],
                                  bm_verts[poly.v3]))
        except ValueError:
            # Duplicate face — skip
            continue

        # Material assignment
        if poly.is_textured:
            resolved = _resolve_tpage(poly.tpage, len(textures))
            key = ('tex', resolved if resolved >= 0 else -1)
        else:
            key = ('rgb', poly.color_r, poly.color_g, poly.color_b)

        if key in mat_index_map:
            face.material_index = mat_index_map[key]

        # UV assignment
        if poly.is_textured:
            if poly.is_overlay:
                # Overlay surface (byte6=0x05): compute projected UVs from vertex positions. 
                # These are flat rectangular surfaces where the stored UVs are invalid (misaligned material read)
                vp1 = model.vertices[poly.v1]
                vp2 = model.vertices[poly.v2]
                vp3 = model.vertices[poly.v3]
                pts = [(vp1.x, vp1.y, vp1.z),
                       (vp2.x, vp2.y, vp2.z),
                       (vp3.x, vp3.y, vp3.z)]
                # Find the 2 axes with most variation for UV projection
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
                rx = max(xs) - min(xs); ry = max(ys) - min(ys); rz = max(zs) - min(zs)
                # Pick the 2 largest axes
                axes = sorted([(rx, 0), (ry, 1), (rz, 2)], reverse=True)
                a0 = axes[0][1]; a1 = axes[1][1]
                vals0 = [p[a0] for p in pts]; vals1 = [p[a1] for p in pts]
                r0 = max(vals0) - min(vals0); r1 = max(vals1) - min(vals1)
                if r0 < 1: r0 = 1
                if r1 < 1: r1 = 1
                for li, vp in enumerate(pts):
                    u = (vp[a0] - min(vals0)) / r0
                    v = (vp[a1] - min(vals1)) / r1
                    face.loops[li][uv_layer].uv = (u, v)
            else:
                face.loops[0][uv_layer].uv = (poly.u0, 1.0 - poly.v0)
                face.loops[1][uv_layer].uv = (poly.u1, 1.0 - poly.v1uv)
                face.loops[2][uv_layer].uv = (poly.u2, 1.0 - poly.v2uv)

    bm.to_mesh(mesh)
    bm.free()

    mesh.update()

    # Store metadata as custom properties
    if model.metadata:
        for key, value in model.metadata.items():
            try:
                obj[f"dfx_{key}"] = value
            except Exception:
                obj[f"dfx_{key}"] = str(value)

    # Create armature if bones are present
    if model.bones and len(model.bones) > 1:
        try:
            _create_armature_for_mesh(obj, model, scale, collection)
        except Exception as e:
            print(f"[DFX] Warning: Failed to create armature for {name}: {e}")

    return obj


def _create_armature_for_mesh(
    mesh_obj: bpy.types.Object,
    model: 'ExtractedModel',
    scale: float,
    collection: bpy.types.Collection,
):
    """Create a Blender armature from DFX bone data and parent the mesh to it."""

    bones = model.bones
    name = mesh_obj.name

    # Create armature data
    arm_data = bpy.data.armatures.new(f"{name}_armature")
    arm_obj = bpy.data.objects.new(f"{name}_armature", arm_data)
    collection.objects.link(arm_obj)

    # Must be in edit mode to add bones
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')

    edit_bones = arm_data.edit_bones

    # Compute world positions for each bone
    bone_world_pos = []
    for i, b in enumerate(bones):
        wx, wy, wz = 0.0, 0.0, 0.0
        aid = i
        safety = 0
        while 0 <= aid < len(bones) and safety < 256:
            wx += float(bones[aid].local_x)
            wy += float(bones[aid].local_y)
            wz += float(bones[aid].local_z)
            pid = bones[aid].parent_id
            if pid == aid or pid == 0xFFFF or pid >= len(bones):
                break
            aid = pid
            safety += 1
        bone_world_pos.append((wx, wy, wz))

    # Create edit bones
    blender_bones = []
    for i, b in enumerate(bones):
        bone_name = f"bone_{i:02d}"
        eb = edit_bones.new(bone_name)

        hx, hy, hz = bone_world_pos[i]
        eb.head = (hx * scale, hy * scale, hz * scale)

        # Bones point along +Y with zero roll so that each bone's local
        # frame equals the armature frame.  The DFX animation channels are
        # per-bone Euler rotations / scales / positions expressed in the
        # parent's frame, so this makes pose-bone transforms map 1:1.
        length = 10 * scale
        for ci, cb in enumerate(bones):
            if cb.parent_id == i and ci != i:
                cx, cy, cz = bone_world_pos[ci]
                d = ((cx - hx) ** 2 + (cy - hy) ** 2 + (cz - hz) ** 2) ** 0.5 * scale
                if d > 0.001:
                    length = d
                break
        eb.tail = (eb.head[0], eb.head[1] + length, eb.head[2])
        eb.roll = 0.0

        blender_bones.append(eb)

    # Set parent relationships (use_connect=False to preserve head positions)
    for i, b in enumerate(bones):
        if b.parent_id < len(bones) and b.parent_id != i and b.parent_id != 0xFFFF:
            blender_bones[i].parent = blender_bones[b.parent_id]
            blender_bones[i].use_connect = False

    bpy.ops.object.mode_set(mode='OBJECT')

    # --- Create vertex groups and assign weights ---
    for i, b in enumerate(bones):
        if b.v_first == 0xFFFF or b.v_last == 0xFFFF:
            continue
        vg = mesh_obj.vertex_groups.new(name=f"bone_{i:02d}")
        last_v = min(b.v_last, len(model.vertices) - 1)
        indices = list(range(b.v_first, last_v + 1))
        if indices:
            vg.add(indices, 1.0, 'REPLACE')

    # Parent mesh to armature with Armature modifier
    mesh_obj.parent = arm_obj
    mod = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
    mod.object = arm_obj




def apply_animations_to_armature(arm_obj, animations, bones=None, scale=0.01):
    """Create one Blender action per parsed animation on the armature.

    Rotation: per-axis Euler, 4096 = 360 deg, X applied first (mode 'XYZ').
    Scale:    1024 = 1.0.
    Position: absolute bone position in parent space -> pose-bone location
              is (value - rest_offset) * scale.
    """
    import math
    if not animations or arm_obj.type != 'ARMATURE':
        return
    for pb in arm_obj.pose.bones:
        pb.rotation_mode = 'XYZ'
    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()

    rest = {}
    if bones:
        for i, b in enumerate(bones):
            rest[i] = (float(b.local_x), float(b.local_y), float(b.local_z))

    def convert(channel, v):
        if channel.plane == ANIM_PLANE_ROT:
            return v / ANIM_ROT_UNIT * 2.0 * math.pi
        if channel.plane == ANIM_PLANE_SCALE:
            return v / ANIM_SCALE_UNIT
        r = rest.get(channel.bone_index, (0.0, 0.0, 0.0))[channel.axis]
        return (v - r) * scale

    path_of = {ANIM_PLANE_ROT: 'rotation_euler',
               ANIM_PLANE_SCALE: 'scale',
               ANIM_PLANE_TRANS: 'location'}

    # Blender 4.4+ removed action.fcurves (slotted actions)
    _use_legacy_fcurves = True
    try:
        test_action = bpy.data.actions.new(name="__dfx_api_test__")
        _ = test_action.fcurves
        bpy.data.actions.remove(test_action)
    except AttributeError:
        _use_legacy_fcurves = False
        try:
            bpy.data.actions.remove(test_action)
        except Exception:
            pass

    created_actions = []
    for anim in animations:
        for w in anim.warnings:
            print(f"[DFX]     '{anim.name}': {w}")
        usable = [c for c in anim.channels
                  if c.keyframes and
                  arm_obj.pose.bones.get(f"bone_{c.bone_index:02d}") is not None]
        skipped = len(anim.channels) - len(usable)
        if not usable:
            print(f"[DFX]     SKIP '{anim.name}': no channels match this rig")
            continue
        kinds = {}
        for c in usable:
            kinds[c.kind] = kinds.get(c.kind, 0) + 1
        planes = sorted({ANIM_PLANE_NAMES[c.plane] for c in usable})
        print(f"[DFX]     '{anim.name}': {len(usable)} ch "
              f"({'+'.join(planes)}, {kinds}), {anim.key_count} keys"
              + (f", {skipped} ch skipped (bone not in mesh)" if skipped else ""))

        action = bpy.data.actions.new(name=f"{arm_obj.name}_{anim.name}")
        action.use_fake_user = True
        action["dfx_frame_count"] = anim.key_count
        action["dfx_channel_count"] = anim.channel_count
        action["dfx_record_addr"] = anim.record_addr

        has_fcurves = False
        if _use_legacy_fcurves:
            for c in usable:
                bone_name = f"bone_{c.bone_index:02d}"
                data_path = f'pose.bones["{bone_name}"].{path_of[c.plane]}'
                fcurve = action.fcurves.new(data_path=data_path, index=c.axis,
                                            action_group=bone_name)
                kps = fcurve.keyframe_points
                kps.add(len(c.keyframes))
                for fi, v in enumerate(c.keyframes):
                    kps[fi].co = (float(fi + 1), convert(c, v))
                    kps[fi].interpolation = 'LINEAR'
                fcurve.update()
                has_fcurves = True
        else:
            arm_obj.animation_data.action = action
            for fi in range(anim.key_count):
                for c in usable:
                    if fi >= len(c.keyframes):
                        continue
                    pb = arm_obj.pose.bones[f"bone_{c.bone_index:02d}"]
                    getattr(pb, path_of[c.plane])[c.axis] = convert(c, c.keyframes[fi])
                    pb.keyframe_insert(data_path=path_of[c.plane], index=c.axis,
                                       frame=fi + 1)
                    has_fcurves = True
            try:
                for fc in action.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = 'LINEAR'
            except Exception:
                pass
            # reset pose so the next action starts from rest
            for pb in arm_obj.pose.bones:
                pb.rotation_euler = (0.0, 0.0, 0.0)
                pb.scale = (1.0, 1.0, 1.0)
                pb.location = (0.0, 0.0, 0.0)

        if has_fcurves:
            created_actions.append((anim, action))
        else:
            bpy.data.actions.remove(action)

    if created_actions:
        _build_bone_anim_playback(arm_obj, created_actions, bones)
    print(f"[DFX] Applied {len(created_actions)} actions to {arm_obj.name}")


# ---- playback planning for bone animations -------------------------------
#
# The files hold no loop/once flag for bone animations: that decision lives
# in the per-object game script.  What the data does tell us is how poses
# connect, so playback is planned from the curves themselves:
#   * an animation whose last pose returns to its first pose loops forever
#   * an animation whose last pose is the first pose of another animation is
#     a transition into it
#   * two animations that are each other's transition form a ping-pong pair
# Anything else is looped as well, since every character seen so far idles

_BONE_POSE_TOL = {0: 60.0, 1: 60.0, 2: 12.0}


def _anim_pose_dicts(anim):
    first, last = {}, {}
    for c in anim.channels:
        k = (c.plane, c.bone_index, c.axis)
        first[k] = c.keyframes[0]
        last[k] = c.keyframes[-1]
    return first, last


def _pose_default(k, bones):
    plane, b, ax = k
    if plane == ANIM_PLANE_ROT:
        return 0.0
    if plane == ANIM_PLANE_SCALE:
        return ANIM_SCALE_UNIT
    if bones and b < len(bones):
        return float((bones[b].local_x, bones[b].local_y, bones[b].local_z)[ax])
    return 0.0


def _pose_match(pa, pb, bones):
    keys = set(pa) | set(pb)
    if not keys:
        return 0.0
    ok = 0
    for k in keys:
        a = pa.get(k, _pose_default(k, bones))
        b = pb.get(k, _pose_default(k, bones))
        diff = abs(a - b)
        if k[0] == ANIM_PLANE_ROT:
            diff = diff % ANIM_ROT_UNIT
            diff = min(diff, ANIM_ROT_UNIT - diff)
        if diff <= _BONE_POSE_TOL[k[0]]:
            ok += 1
    return ok / len(keys)


def _anim_self_loops(anim):
    tot = closed = 0
    for c in anim.channels:
        v = c.keyframes
        rng = max(v) - min(v)
        if rng < 8:
            continue
        tot += 1
        if abs(v[-1] - v[0]) <= 0.15 * rng:
            closed += 1
    return tot == 0 or closed / tot >= 0.5


def _plan_bone_playback(anims, bones):
    """Return (loops, successor) lists: loops[i] True if anim i closes on
    itself, successor[i] = index of the anim it transitions into or None."""
    P = [_anim_pose_dicts(a) for a in anims]
    loops, succ = [], []
    for i, a in enumerate(anims):
        if _anim_self_loops(a):
            loops.append(True)
            succ.append(None)
            continue
        loops.append(False)
        best, bs = None, 0.0
        for j in range(len(anims)):
            if j == i:
                continue
            sc = _pose_match(P[i][1], P[j][0], bones)
            if sc > bs:
                bs, best = sc, j
        succ.append(best if best is not None and bs >= 0.8 else None)
    return loops, succ


def _add_cycles(action, target, mode='REPEAT'):
    for fc in _action_fcurves(action, target):
        if len(fc.keyframe_points) > 1 and not any(m.type == 'CYCLES' for m in fc.modifiers):
            m = fc.modifiers.new('CYCLES')
            m.mode_before = mode
            m.mode_after = mode


def _nla_strip(track, action, start, name=None):
    strip = track.strips.new(name or action.name, int(start), action)
    try:
        strip.action_frame_end = float(action.get("dfx_frame_count", strip.action_frame_end))
    except Exception:
        pass
    try:
        if getattr(strip, "action_slot", None) is None and len(action.slots):
            strip.action_slot = action.slots[0]
    except Exception:
        pass
    return strip


def _build_bone_anim_playback(arm_obj, created, bones, target_frames=3000):
    anims = [a for a, _ in created]
    actions = [act for _, act in created]
    loops, succ = _plan_bone_playback(anims, bones)
    ad = arm_obj.animation_data

    # walk the chain from the primary animation
    chain, seen = [], set()
    i = 0
    while i is not None and i not in seen:
        seen.add(i)
        chain.append(i)
        if loops[i]:
            break
        i = succ[i]
    cycle_start = chain.index(i) if (i is not None and i in seen and not loops[i]) else None

    for idx, act in enumerate(actions):
        if loops[idx]:
            act["dfx_play_mode"] = 'loop'
        elif succ[idx] is not None:
            act["dfx_play_mode"] = f'then {anims[succ[idx]].name}'
        else:
            act["dfx_play_mode"] = 'loop'

    in_sequence = set(chain)
    if len(chain) == 1 and cycle_start is None:
        # plain loop: active action + cycles modifier
        ad.action = actions[0]
        _add_cycles(actions[0], arm_obj)
        print(f"[DFX]     playback: '{anims[0].name}' loops")
    else:
        ad.action = None
        track = ad.nla_tracks.new()
        track.name = "sequence"
        frame = 1
        order = list(chain)
        if cycle_start is not None:
            head, cyc = order[:cycle_start], order[cycle_start:]
            seq = head[:]
            while frame + sum(anims[k].key_count for k in seq) < target_frames and len(seq) < 60:
                seq += cyc
            order = seq
        for n, k in enumerate(order):
            strip = _nla_strip(track, actions[k], frame, f"{anims[k].name}_{n:02d}")
            ln = max(1, anims[k].key_count)
            if n == len(order) - 1 and loops[k]:
                try:
                    strip.repeat = max(1.0, float(target_frames - frame) / ln)
                except Exception:
                    pass
            frame += ln
        desc = ' -> '.join(anims[k].name for k in chain) + (' (ping-pong)' if cycle_start is not None else ' (loop)')
        print(f"[DFX]     playback: {desc}")

    for idx, act in enumerate(actions):
        if idx in in_sequence:
            continue
        track = ad.nla_tracks.new()
        track.name = act.name
        _nla_strip(track, act, 1)
        track.mute = True
        if loops[idx]:
            _add_cycles(act, arm_obj)



def load_textures_to_blender(
    vd3_data: bytes,
    textures: List[VD3Texture],
    tex_dir: str,
) -> dict:
    """Export VD3 textures as PNG files and load them as Blender images.
    Returns dict mapping texture index → bpy.types.Image."""

    os.makedirs(tex_dir, exist_ok=True)
    tex_images = {}

    for tex in textures:
        try:
            png_path = os.path.join(tex_dir, tex.png_filename)
            if not os.path.exists(png_path):
                save_texture_png(vd3_data, tex, png_path)

            img = bpy.data.images.load(png_path)
            img.alpha_mode = 'STRAIGHT'
            tex_images[tex.index] = img
        except Exception as e:
            print(f"[DFX] Warning: Failed to load texture {tex.index} '{tex.tag}': {e}")

    return tex_images

# ========================================================
#  Level animation extraction (data section +0x20 / +0x24)
# ========================================================

def extract_level_anim_table(data):
    """Extract level animation table entries from data section +0x20/+0x24.
    Each entry has position vertices and triangle mesh connectivity."""
    seg_count = ru32(data, 0x20)
    seg_ptr = ru32(data, 0x24)
    if seg_count == 0 or seg_ptr == 0 or seg_ptr >= len(data) or seg_count > 500:
        return []
    
    # Pre-scan: compute global valid sector_ptr range across all declared sky tris
    sky_sptr_min = 0xFFFFFFFF
    sky_sptr_max = 0
    for si in range(seg_count):
        hoff = seg_ptr + si * 24
        if hoff + 24 > len(data): break
        mp = ru32(data, hoff + 8)
        tc = ri16(data, hoff + 0x14)
        vc = ru16(data, hoff + 2)
        if mp == 0 or tc <= 0: continue
        for ti in range(tc):
            toff = mp + ti * 12
            if toff + 12 > len(data): break
            v0 = ru16(data, toff); v1 = ru16(data, toff+2); v2 = ru16(data, toff+4)
            sptr = ru32(data, toff + 8)
            if max(v0, v1, v2) < 5000 and 0x1000 < sptr < len(data):
                sky_sptr_min = min(sky_sptr_min, sptr)
                sky_sptr_max = max(sky_sptr_max, sptr)
    if sky_sptr_min > sky_sptr_max:
        sky_sptr_min = 0; sky_sptr_max = len(data)
    
    segments = []
    for si in range(seg_count):
        hoff = seg_ptr + si * 24
        if hoff + 24 > len(data): break
        stype = ru16(data, hoff)
        vert_count = ru16(data, hoff + 2)
        vert_ptr = ru32(data, hoff + 4)
        mesh_ptr = ru32(data, hoff + 8)
        tri_count = ri16(data, hoff + 0x14)
        if vert_count == 0 or vert_ptr == 0 or vert_ptr >= len(data): continue
        meta = [ri16(data, hoff + 12 + i * 2) for i in range(6)]
        
        # First pass: find valid sector_ptr range from declared tris,
        # then scan past declared count for extra valid tris
        real_vc = vert_count
        real_tc = 0
        if mesh_ptr > 0 and mesh_ptr < len(data) and tri_count > 0:
            # Now scan including extras
            for ti in range(max(tri_count * 2, tri_count + 200)):
                toff = mesh_ptr + ti * 12
                if toff + 12 > len(data): break
                v0 = ru16(data, toff); v1 = ru16(data, toff+2); v2 = ru16(data, toff+4)
                pad = ru16(data, toff + 6)
                sptr = ru32(data, toff + 8)
                max_vi = max(v0, v1, v2)
                if max_vi > 5000:
                    break
                # For tris past declared count, require pad=0 AND valid sector_ptr
                if ti >= tri_count:
                    if pad != 0 or sptr < sky_sptr_min or sptr > sky_sptr_max:
                        break
                real_tc = ti + 1
                if max_vi >= real_vc:
                    real_vc = max_vi + 1
        
        # Read vertices up to the real count (may exceed declared)
        verts = []
        for vi in range(real_vc):
            voff = vert_ptr + vi * 8
            if voff + 6 > len(data): break
            x = ri16(data, voff); y = ri16(data, voff + 2); z = ri16(data, voff + 4)
            verts.append((float(x), float(y), float(z)))
        
        # Read triangle connectivity (12 bytes each: u16 v0,v1,v2, u16 pad, u32 sector_ptr)
        # sector_ptr points to material data: same format as level geo _read_material
        tris = []
        tri_mats = []
        if mesh_ptr > 0 and mesh_ptr < len(data) and real_tc > 0:
            for ti in range(real_tc):
                toff = mesh_ptr + ti * 12
                if toff + 12 > len(data): break
                v0 = ru16(data, toff)
                v1 = ru16(data, toff + 2)
                v2 = ru16(data, toff + 4)
                sptr = ru32(data, toff + 8)
                if v0 < len(verts) and v1 < len(verts) and v2 < len(verts):
                    tris.append((v0, v1, v2))
                    # Read material from sector_ptr (same layout as _read_material)
                    if 0 < sptr < len(data) - 10:
                        u0 = data[sptr] / 255.0
                        v0uv = data[sptr + 1] / 255.0
                        sky_clut = ru16(data, sptr + 2)
                        u1 = data[sptr + 4] / 255.0
                        v1uv = data[sptr + 5] / 255.0
                        tpage = ru16(data, sptr + 6)
                        u2 = data[sptr + 8] / 255.0
                        v2uv = data[sptr + 9] / 255.0
                        tri_mats.append((tpage, u0, v0uv, u1, v1uv, u2, v2uv, sky_clut))
                    else:
                        tri_mats.append(None)
        
        if verts:
            seg = SplineSegment()
            seg.seg_type = stype
            seg.vertices = verts
            seg.triangles = tris
            seg.tri_materials = tri_mats
            seg.meta = meta
            segments.append(seg)
    return segments

def extract_level_anim_flat(data):
    segments = extract_level_anim_table(data)
    waypoints = []
    for seg in segments: waypoints.extend(seg.vertices)
    return waypoints


# =====================================================================

@dataclass
class PlacementRecord:
    """One instance placement record from the level instance table."""
    obj_ptr: int = 0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rotation_raw: int = 0      # i16 at +12, 4096 = 360°
    flags: int = 0             # u16 at +14
    draw_distance: int = 0
    anim_rec: dict = None  # parsed per-instance animation record
    obj_flags: int = 0         # object header flags (obj+0x00)
    tilt_x: int = 0            # +0x08 i16: pitch (4096 = 360deg)
    tilt_y: int = 0            # +0x0A i16: roll  (4096 = 360deg)
    param_ptr: int = 0         # +0x04 u32: optional per-object parameter block
    intro_dist: int = 0        # object intro radius (i16 at obj+0x1A): the engine
                               # spawns the instance when the camera comes within
                               # this 3D distance of the placement position


def extract_placement_table(data: bytes) -> List[PlacementRecord]:
    """Extract all object placement records from the level instance table.
    Located via level geometry header: +0x10 = count, +0x14 = table pointer.
    Each record is 0x30 (48) bytes.

    Record layout:
      +0x00  u32   object definition pointer
      +0x04  u32   (flags, usually 0)
      +0x08  u32   (flags, usually 0)
      +0x0C  i16   rotation angle (4096 = full circle)
      +0x0E  u16   flags (0/4/5/6)
      +0x10  i16   X position (world space)
      +0x12  i16   Y position (world space)
      +0x14  i16   Z position (height)
      +0x16  i16   draw distance / scale
    """

    lp = ru32(data, 0)
    if lp + 0x18 > len(data):
        return []

    inst_count = ru32(data, lp + 0x10)
    inst_ptr = ru32(data, lp + 0x14)

    if inst_count == 0 or inst_count > 10000:
        return []
    if inst_ptr == 0 or inst_ptr + inst_count * 0x30 > len(data):
        return []

    records = []
    for i in range(inst_count):
        roff = inst_ptr + i * 0x30
        if roff + 0x30 > len(data):
            break

        rec = PlacementRecord()
        rec.obj_ptr = ru32(data, roff)
        rec.x = float(ri16(data, roff + 16))
        rec.y = float(ri16(data, roff + 18))
        rec.z = float(ri16(data, roff + 20))     # Z height at +20
        rec.rotation_raw = ri16(data, roff + 12)  # Rotation at +12
        rec.flags = ru16(data, roff + 14)
        rec.draw_distance = ri16(data, roff + 22)
        # Per-instance animation record pointer at +0x24
        rec.param_ptr = ru32(data, roff + 0x04)
        rec.tilt_x = ri16(data, roff + 0x08)
        rec.tilt_y = ri16(data, roff + 0x0A)
        anim_ptr = ru32(data, roff + 0x24)
        if 0 < rec.obj_ptr < len(data) - 4:
            rec.obj_flags = ru32(data, rec.obj_ptr)
            rec.intro_dist = ri16(data, rec.obj_ptr + 0x1A)
        if anim_ptr > 0 and anim_ptr < len(data):
            rec.anim_rec = parse_instance_anim_record(data, anim_ptr, rec.obj_flags)
        else:
            rec.anim_rec = None
        records.append(rec)

    return records




def create_level_anim_objects(segments, scale, collection, data=None,
                              textures=None, tex_images=None, mat_cache=None):
    """Create Blender mesh objects from skybox table entries.
    Each entry has vertices, triangle connectivity, and per-triangle materials."""
    created = []
    if textures is None:
        textures = []
    if tex_images is None:
        tex_images = {}
    if mat_cache is None:
        mat_cache = {}

    for si, seg in enumerate(segments):
        if not seg.vertices:
            continue

        mesh_name = f"sky_{si:02d}_t{seg.seg_type}"
        mesh = bpy.data.meshes.new(mesh_name)
        obj = bpy.data.objects.new(mesh_name, mesh)

        # Sky vertices are stored at 1/16 scale relative to level geometry
        sky_scale = scale * 16.0
        verts_co = [(x * sky_scale, y * sky_scale, z * sky_scale)
                     for x, y, z in seg.vertices]

        faces = []
        if seg.triangles:
            for v0, v1, v2 in seg.triangles:
                if (v0 < len(verts_co) and v1 < len(verts_co) and
                        v2 < len(verts_co)):
                    faces.append((v0, v1, v2))

        if faces:
            mesh.from_pydata(verts_co, [], faces)
        else:
            mesh.from_pydata(verts_co, [], [])
        mesh.update()

        # Apply per-triangle textures and UVs if material data is available
        has_mats = (seg.tri_materials and textures and
                    len(seg.tri_materials) == len(faces))
        if has_mats:
            uv_layer = mesh.uv_layers.new(name="UVMap")

            # Group triangles by tpage → material
            tpage_to_mat_idx = {}
            for fi, tmat in enumerate(seg.tri_materials):
                if tmat is None:
                    continue
                tpage = tmat[0]
                resolved = _resolve_tpage(tpage, len(textures))
                if resolved < 0:
                    continue

                if resolved not in tpage_to_mat_idx:
                    mat = _get_or_create_material_textured(
                        resolved, textures, tex_images, mat_cache)
                    if mat:
                        obj.data.materials.append(mat)
                        tpage_to_mat_idx[resolved] = len(obj.data.materials) - 1

            # Assign materials and UVs per face
            for fi, face in enumerate(mesh.polygons):
                if fi >= len(seg.tri_materials):
                    break
                tmat = seg.tri_materials[fi]
                if tmat is None:
                    continue

                tpage, u0, v0, u1, v1, u2, v2 = tmat[0], tmat[1], tmat[2], tmat[3], tmat[4], tmat[5], tmat[6]
                resolved = _resolve_tpage(tpage, len(textures))

                if resolved in tpage_to_mat_idx:
                    face.material_index = tpage_to_mat_idx[resolved]

                # Set UVs for each loop of this face
                for li, loop_idx in enumerate(face.loop_indices):
                    if li == 0:
                        uv_layer.data[loop_idx].uv = (u0, 1.0 - v0)
                    elif li == 1:
                        uv_layer.data[loop_idx].uv = (u1, 1.0 - v1)
                    elif li == 2:
                        uv_layer.data[loop_idx].uv = (u2, 1.0 - v2)

        # Store metadata
        obj["dfx_sky_type"] = seg.seg_type
        obj["dfx_sky_verts"] = len(seg.vertices)
        obj["dfx_sky_tris"] = len(faces)

        collection.objects.link(obj)
        created.append(obj)
    return created

def create_level_anim_curve(waypoints, scale, collection):
    if not waypoints: return None
    curve_data = bpy.data.curves.new("level_anim", type='CURVE')
    curve_data.dimensions = '3D'; curve_data.resolution_u = 12
    spline = curve_data.splines.new('NURBS')
    spline.points.add(len(waypoints) - 1)
    for i, (x, y, z) in enumerate(waypoints):
        spline.points[i].co = (x * scale, y * scale, z * scale, 1.0)
    spline.use_cyclic_u = True; spline.use_endpoint_u = False
    obj = bpy.data.objects.new("level_anim", curve_data)
    collection.objects.link(obj); return obj



# =========================================================================
#  Per-instance animation records (position / rotation / scale tracks)
# =========================================================================
#
# Placement record +0x24 points to an animation record: an array of track
# pointers, slot 0 = position, slot 1 = rotation, slot 2 = scale (a slot
# may be 0).  Each track header is 8 bytes:
#     +0 u32 keys_ptr, +4 s16 key_count, +6 u8 kind (0/1/2), +7 u8 flags
# flags: 0x04 = loop (restart at key 0; the final key is only the
#        interpolation target of the previous one), 0x01 = play once.
#
# Position / scale keys are 40 bytes:
#     +0 u16 duration in frames until the next key
#     +2 i16 x, +4 i16 y, +6 i16 z   (absolute world units / 4096 = 1.0 scale)
#     +8  i32 out_tangent[3], +20 i32 in_tangent[3], +32 8 bytes unused
#   and are interpolated with a cubic Hermite:
#     P(t) = h00*P0 + h10*out0 + h01*P1 + h11*in1
# Rotation keys are 10 bytes:
#     +0 u16 duration, +2 i16 qx, qy, qz, qw (4096 = 1.0), slerped.
#
# When a non-looping track reaches its end the engine reverses playback
# (ping-pong) unless the object header flags (obj+0) contain 0x8000 or
# 0x80000 (stop at end), 0x20000 (one-shot) or 0x40000 (no reverse).

INST_TRACK_POS, INST_TRACK_ROT, INST_TRACK_SCALE = 0, 1, 2
INST_TRACK_LOOP = 0x04


def _read_track(data, hdr_addr):
    if not (0 < hdr_addr < len(data) - 8):
        return None
    keys_ptr = ru32(data, hdr_addr)
    count = ri16(data, hdr_addr + 4)
    kind = data[hdr_addr + 6]
    flags = data[hdr_addr + 7]
    if count <= 0 or count > 4096 or kind > 2:
        return None
    if not (0 < keys_ptr < len(data)):
        return None
    keys = []
    if kind == INST_TRACK_ROT:
        for i in range(count):
            o = keys_ptr + i * 10
            if o + 10 > len(data):
                break
            keys.append({
                'dur': ru16(data, o),
                'quat': (ri16(data, o + 2), ri16(data, o + 4),
                         ri16(data, o + 6), ri16(data, o + 8)),
            })
    else:
        for i in range(count):
            o = keys_ptr + i * 40
            if o + 40 > len(data):
                break
            keys.append({
                'dur': ru16(data, o),
                'pos': (ri16(data, o + 2), ri16(data, o + 4), ri16(data, o + 6)),
                'out': (ri32(data, o + 8), ri32(data, o + 12), ri32(data, o + 16)),
                'in': (ri32(data, o + 20), ri32(data, o + 24), ri32(data, o + 28)),
            })
    return {'addr': hdr_addr, 'kind': kind, 'flags': flags,
            'loop': bool(flags & INST_TRACK_LOOP), 'keys': keys}


def parse_instance_anim_record(data, rec_addr, obj_flags=0):
    """Parse the animation record of a placement. Returns a dict with the
    three tracks (may be None) and the playback mode, or None."""
    if rec_addr <= 0 or rec_addr + 12 > len(data):
        return None
    tracks = {}
    for slot, name in ((0, 'pos'), (1, 'rot'), (2, 'scale')):
        hdr = ru32(data, rec_addr + slot * 4)
        tracks[name] = _read_track(data, hdr) if hdr else None
    if not any(tracks.values()):
        return None
    # End-of-track behaviour from the engine's instance updater: a finished
    # non-looping track reverses direction (ping-pong) unless the object
    # flags say otherwise: 0x8000/0x80000 deactivate at the end, 0x20000
    # freezes, 0x10000 kills the instance, 0x40000 suppresses the reverse.
    STOP_FLAGS = 0x8000 | 0x80000 | 0x20000 | 0x10000 | 0x40000
    loop = any(t and t['loop'] for t in tracks.values())
    if loop:
        mode = 'loop'
    elif obj_flags & STOP_FLAGS:
        mode = 'once'
    else:
        mode = 'pingpong'
    pos = tracks['pos']
    waypoints = [tuple(float(v) for v in k['pos']) for k in pos['keys']] if pos else []
    return {
        'addr': rec_addr,
        'tracks': tracks,
        'mode': mode,
        'waypoints': waypoints,
        'type': 'track',
    }


def _track_frame_times(keys, offset=0.0):
    """Cumulative start frame of every key (frame 1 + offset = first key)."""
    times = []
    f = 0.0
    for k in keys:
        times.append(1.0 + offset + f)
        f += k['dur']
    return times


def _action_fcurves(action, target):
    """F-curves of an action, for both legacy and slotted (4.4+) actions."""
    try:
        return list(action.fcurves)
    except AttributeError:
        pass
    try:
        slot = target.animation_data.action_slot
        for layer in action.layers:
            for strip in layer.strips:
                cb = strip.channelbag(slot)
                if cb is not None:
                    return list(cb.fcurves)
    except Exception:
        pass
    return []


def _create_track_path_curve(name, track, scale, cyclic, collection):
    """Bezier curve through the position keys, handles from the Hermite
    tangents so the curve shows the exact path the engine interpolates."""
    keys = track['keys']
    if len(keys) < 2:
        return None
    curve_data = bpy.data.curves.new(name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 12
    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(len(keys) - 1)
    for i, k in enumerate(keys):
        bp = spline.bezier_points[i]
        p = [v * scale for v in k['pos']]
        bp.co = p
        bp.handle_left_type = 'FREE'
        bp.handle_right_type = 'FREE'
        bp.handle_right = [p[a] + k['out'][a] * scale / 3.0 for a in range(3)]
        bp.handle_left = [p[a] - k['in'][a] * scale / 3.0 for a in range(3)]
    spline.use_cyclic_u = cyclic
    path_obj = bpy.data.objects.new(name, curve_data)
    path_obj["dfx_path_points"] = len(keys)
    path_obj["dfx_path_loop"] = track['loop']
    path_obj.display_type = 'WIRE'
    path_obj.show_in_front = True
    collection.objects.link(path_obj)
    return path_obj


# ---- instance spawn timing (reverse-engineered from the game code) -------
#
# The engine streams instances: INSTANCE_IntroduceInstance runs when the
# camera focus point comes within the object's intro radius (i16 at
# obj+0x1A) of the placement position (3D distance), and the instance is
# removed again beyond obj+0x1C.  Track state (anim record +0x10) is
# zero-initialised in the file, so a track starts playing the moment its
# instance spawns.  During the level intro the focus point follows the
# flyover camera track

def _camera_position_samples(placements):
    """Per-frame (x, y, z) of the level's scripted flyover camera, or None."""
    for rec in placements:
        if rec.obj_ptr != 0 or rec.anim_rec is None:
            continue
        pos = rec.anim_rec['tracks'].get('pos')
        if not pos or len(pos['keys']) < 2:
            continue
        samples = []
        keys = pos['keys']
        for i in range(len(keys) - 1):
            a, b = keys[i], keys[i + 1]
            dur = max(1, a['dur'])
            for t in range(dur):
                u = t / dur
                samples.append(tuple(a['pos'][k] + (b['pos'][k] - a['pos'][k]) * u
                                     for k in range(3)))
        samples.append(tuple(keys[-1]['pos']))
        return samples
    return None


def _instance_spawn_frame(rec, cam_samples):
    """First flyover frame at which the engine spawns this instance: the
    camera within the object's intro radius of the placement (3D).
    Returns 0 if spawned from the start, None if never during the intro."""
    if not cam_samples or rec.intro_dist <= 0:
        return 0
    r2 = rec.intro_dist * rec.intro_dist
    for f, (cx, cy, cz) in enumerate(cam_samples):
        dx, dy, dz = cx - rec.x, cy - rec.y, cz - rec.z
        if dx * dx + dy * dy + dz * dz < r2:
            return f
    return None


def _shift_nla_sequence(anim_data, offset):
    """Move every strip of unmuted NLA tracks later by `offset` frames.
    Strips are rebuilt at the shifted positions because frame_start_ui /
    frame_end_ui are resize handles and do not reliably translate strips."""
    if offset <= 0:
        return
    for track in anim_data.nla_tracks:
        if track.mute:
            continue
        specs = []
        for strip in track.strips:
            specs.append((
                strip.name,
                strip.frame_start,
                strip.action,
                getattr(strip, "action_frame_start", None),
                strip.action_frame_end,
                getattr(strip, "repeat", 1.0),
                getattr(strip, "action_slot", None),
            ))
        for strip in list(track.strips):
            track.strips.remove(strip)
        for name, fs, action, afs, afe, rep, slot in specs:
            if action is None:
                continue
            strip = track.strips.new(name, int(fs + offset), action)
            try:
                if afs is not None:
                    strip.action_frame_start = afs
                strip.action_frame_end = afe
                strip.repeat = rep
            except Exception:
                pass
            try:
                if slot is not None:
                    strip.action_slot = slot
            except Exception:
                pass


def apply_instance_tracks(target, anim, scale, name, frame_offset=0):
    """Keyframe an instance object from its position / rotation / scale
    tracks. One game tick = one Blender frame, starting at frame
    1 + frame_offset. If the object already has animation (a bone-animated
    armature), the track goes onto an NLA strip so both play together."""
    tracks = anim['tracks']
    mode = anim['mode']
    if target.animation_data is None:
        target.animation_data_create()
    prev_action = target.animation_data.action
    had_anim = prev_action is not None or len(target.animation_data.nla_tracks) > 0
    action = bpy.data.actions.new(name=f"{name}_track")
    action.use_fake_user = True
    target.animation_data.action = action
    action["dfx_play_mode"] = mode
    fcurve_meta = {}   # data_path -> per-key (frame, handles) for Bezier

    pos = tracks.get('pos')
    if pos and pos['keys']:
        keys = pos['keys']
        times = _track_frame_times(keys, frame_offset)
        hl = {}
        for i, k in enumerate(keys):
            f = times[i]
            target.location = tuple(v * scale for v in k['pos'])
            target.keyframe_insert(data_path="location", frame=f)
            d_next = keys[i]['dur'] if i < len(keys) - 1 else (keys[i - 1]['dur'] if i else 1)
            d_prev = keys[i - 1]['dur'] if i else d_next
            p = [v * scale for v in k['pos']]
            hl[i] = (
                f,
                [(f - d_prev / 3.0, p[a] - k['in'][a] * scale / 3.0) for a in range(3)],
                [(f + d_next / 3.0, p[a] + k['out'][a] * scale / 3.0) for a in range(3)],
            )
        fcurve_meta["location"] = hl
        action["dfx_pos_keys"] = len(keys)

    rot = tracks.get('rot')
    if rot and rot['keys']:
        keys = rot['keys']
        times = _track_frame_times(keys, frame_offset)
        target.rotation_mode = 'QUATERNION'
        prev = None
        for i, k in enumerate(keys):
            x, y, z, w = (v / 4096.0 for v in k['quat'])
            q = [w, x, y, z]
            n = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) ** 0.5 or 1.0
            q = [v / n for v in q]
            if prev is not None and sum(a * b for a, b in zip(q, prev)) < 0:
                q = [-v for v in q]
            prev = q
            target.rotation_quaternion = q
            target.keyframe_insert(data_path="rotation_quaternion", frame=times[i])
        action["dfx_rot_keys"] = len(keys)

    scl = tracks.get('scale')
    if scl and scl['keys']:
        keys = scl['keys']
        times = _track_frame_times(keys, frame_offset)
        for i, k in enumerate(keys):
            target.scale = tuple(v / 4096.0 for v in k['pos'])
            target.keyframe_insert(data_path="scale", frame=times[i])

    total = 0
    main = pos or rot or scl
    if main:
        total = sum(k['dur'] for k in main['keys'])
    action["dfx_total_frames"] = total

    for fc in _action_fcurves(action, target):
        meta = fcurve_meta.get(fc.data_path)
        for i, kp in enumerate(fc.keyframe_points):
            if meta is not None and i in meta:
                f, left, right = meta[i]
                kp.interpolation = 'BEZIER'
                kp.handle_left_type = 'FREE'
                kp.handle_right_type = 'FREE'
                kp.handle_left = left[fc.array_index]
                kp.handle_right = right[fc.array_index]
            else:
                kp.interpolation = 'LINEAR'
        if mode in ('loop', 'pingpong') and len(fc.keyframe_points) > 1:
            mod = fc.modifiers.new('CYCLES')
            mod.mode_after = 'REPEAT' if mode == 'loop' else 'MIRROR'
            mod.mode_before = 'REPEAT' if mode == 'loop' else 'NONE'
        fc.update()

    if had_anim:
        # The armature already plays a bone animation: keep it and put the
        # object-level track on its own NLA strip instead.
        target.animation_data.action = prev_action
        track = target.animation_data.nla_tracks.new()
        track.name = f"{name}_path"
        strip = track.strips.new(action.name, int(1 + frame_offset), action)
        try:
            if getattr(strip, "action_slot", None) is None and len(action.slots):
                strip.action_slot = action.slots[0]
        except Exception:
            pass
        if mode == 'loop' and total > 0:
            try:
                strip.repeat = max(1.0, 3000.0 / total)
            except Exception:
                pass
    return action


def apply_placements(
    placements: List[PlacementRecord],
    obj_ptr_to_blender: dict,
    scale: float,
    collection: bpy.types.Collection,
):
    """Create duplicates of objects at their placement positions
    with rotation applied. Each instance gets its own armature copy."""

    import math
    from collections import Counter
    ptr_counter = Counter()

    cam_samples = _camera_position_samples(placements)

    for rec in placements:
        # Handle null-model instances (camera paths, sound emitters, triggers)
        if rec.obj_ptr not in obj_ptr_to_blender:
            if rec.obj_ptr == 0 and rec.anim_rec is not None:
                # Null-model instance with tracks: the level's scripted
                # camera. Build an Empty that follows the tracks and a
                # camera child looking along the Empty's +Y axis.
                ar = rec.anim_rec
                if ar['mode'] == 'pingpong':
                    ar['mode'] = 'once'   # the flyover plays once, then the race starts
                ptr_counter[0] += 1
                cam_name = f"camera_{ptr_counter[0]:02d}"
                empty = bpy.data.objects.new(cam_name, None)
                empty.empty_display_type = 'ARROWS'
                empty.empty_display_size = 50 * scale
                empty.location = (rec.x * scale, rec.y * scale, rec.z * scale)
                empty["dfx_null_model"] = True
                empty["dfx_anim_mode"] = ar['mode']
                collection.objects.link(empty)
                cam_data = bpy.data.cameras.new(cam_name)
                cam_data.clip_end = 100000 * scale
                cam_obj = bpy.data.objects.new(f"{cam_name}_view", cam_data)
                cam_obj.parent = empty
                cam_obj.rotation_euler = (math.pi / 2.0, 0.0, 0.0)
                collection.objects.link(cam_obj)
                pos = ar['tracks'].get('pos')
                if pos:
                    _create_track_path_curve(f"{cam_name}_path", pos, scale,
                                             ar['mode'] == 'loop', collection)
                try:
                    apply_instance_tracks(empty, ar, scale, cam_name)
                except Exception as e:
                    print(f"[DFX]   WARNING: camera track failed: {e}")
            continue

        source_obj = obj_ptr_to_blender[rec.obj_ptr]
        ptr_counter[rec.obj_ptr] += 1
        inst_idx = ptr_counter[rec.obj_ptr]

        inst_name = f"{source_obj.name}_inst{inst_idx:02d}"

        # Rotation
        rot_rad = (rec.rotation_raw / 4096.0) * 2.0 * math.pi

        # Check if source has an armature parent
        source_arm = source_obj.parent if (
            source_obj.parent and source_obj.parent.type == 'ARMATURE') else None

        if source_arm:
            # Duplicate the armature (full copy with its own data)
            new_arm = source_arm.copy()
            new_arm.data = source_arm.data.copy()
            new_arm.name = f"{inst_name}_armature"
            new_arm.location = (rec.x * scale, rec.y * scale, rec.z * scale)
            tilt = (rec.tilt_x / 4096.0 * 2 * math.pi,
                    rec.tilt_y / 4096.0 * 2 * math.pi)
            new_arm.rotation_euler = (tilt[0], tilt[1], rot_rad)
            collection.objects.link(new_arm)

            # Spawn timing: shift one-shot sequences to the frame the
            # engine spawns this instance during the intro flyover
            _spawn = 0
            if cam_samples is not None and source_arm.animation_data:
                _has_seq = any((not t.mute) and t.name == "sequence"
                               for t in source_arm.animation_data.nla_tracks)
                if _has_seq:
                    _spawn = _instance_spawn_frame(rec, cam_samples) or 0

            # Deep-copy animation_data (NLA tracks, strips, actions)
            if source_arm.animation_data:
                if new_arm.animation_data is None:
                    new_arm.animation_data_create()
                if source_arm.animation_data.action:
                    new_arm.animation_data.action = source_arm.animation_data.action
                for src_track in source_arm.animation_data.nla_tracks:
                    dst_track = new_arm.animation_data.nla_tracks.new()
                    dst_track.name = src_track.name
                    dst_track.mute = src_track.mute
                    for src_strip in src_track.strips:
                        if src_strip.action:
                            dst_strip = dst_track.strips.new(
                                src_strip.name,
                                int(src_strip.frame_start),
                                src_strip.action)
                            dst_strip.action_frame_end = src_strip.action_frame_end
                            try:
                                dst_strip.repeat = src_strip.repeat
                                dst_strip.action_frame_start = src_strip.action_frame_start
                            except Exception:
                                pass
                            try:
                                if getattr(src_strip, "action_slot", None) is not None:
                                    dst_strip.action_slot = src_strip.action_slot
                            except Exception:
                                pass

            if _spawn:
                _shift_nla_sequence(new_arm.animation_data, _spawn)
                new_arm["dfx_spawn_frame"] = _spawn
                print(f"[DFX]     {inst_name}: spawns at intro frame {_spawn + 1} "
                      f"(camera within {rec.intro_dist} units)")

            # Duplicate the mesh (linked = shares mesh data)
            new_obj = source_obj.copy()
            new_obj.name = inst_name
            new_obj.parent = new_arm
            # Reset the local transform since position comes from armature
            new_obj.location = (0, 0, 0)
            new_obj.rotation_euler = (0, 0, 0)

            # Update armature modifier to point to the new armature
            for mod in new_obj.modifiers:
                if mod.type == 'ARMATURE':
                    mod.object = new_arm

            collection.objects.link(new_obj)
        else:
            # No armature — simple linked duplicate
            new_obj = source_obj.copy()
            new_obj.name = inst_name
            new_obj.location = (rec.x * scale, rec.y * scale, rec.z * scale)
            tilt = (rec.tilt_x / 4096.0 * 2 * math.pi,
                    rec.tilt_y / 4096.0 * 2 * math.pi)
            new_obj.rotation_euler = (tilt[0], tilt[1], rot_rad)
            collection.objects.link(new_obj)

        # Store placement metadata
        new_obj["dfx_placement"] = True
        new_obj["dfx_rotation_deg"] = round((rec.rotation_raw / 4096.0) * 360.0, 1)
        new_obj["dfx_rotation_raw"] = rec.rotation_raw
        new_obj["dfx_flags"] = rec.flags
        new_obj["dfx_draw_distance"] = rec.draw_distance

        if rec.param_ptr:
            new_obj["dfx_param_ptr"] = rec.param_ptr

        # Runtime FX emitters (water droplets/laps/falls, fireworks...):
        # their meshes are engine placeholder boxes; the visible effect is a
        # particle system generated per-frame by the FX module. Hide the box.
        _fx_base = source_obj.name.split('_', 1)[-1].lower()
        if _fx_base.startswith('xwtr'):
            new_obj.display_type = 'WIRE'
            new_obj.hide_render = True
            new_obj["dfx_fx_emitter"] = True
            if 'new_arm' in dir() and source_arm and new_arm:
                new_arm["dfx_fx_emitter"] = True

        # Per-instance tracks (position / rotation / scale)
        if rec.anim_rec is not None:
            ar = rec.anim_rec
            new_obj["dfx_anim_mode"] = ar['mode']
            target = new_arm if source_arm else new_obj
            base_name = source_obj.name.split('_', 1)[-1].lower()
            is_ai = bool(rec.obj_flags & 0x800) or base_name in ('aimain', 'aialt', 'aiintro')
            pos = ar['tracks'].get('pos')
            if pos:
                cyclic = ar['mode'] == 'loop' or (is_ai and 'intro' not in source_obj.name)
                _create_track_path_curve(f"{inst_name}_path", pos, scale, cyclic, collection)
                new_obj["dfx_path_waypoints"] = len(pos['keys'])
            if is_ai:
                # AI racing lines: data for the CPU karts, nothing to move
                new_obj["dfx_ai_path"] = True
            else:
                spawn = 0
                if cam_samples is not None and ar['mode'] in ('once', 'pingpong'):
                    # one-shot tracks start when the engine spawns the
                    # instance: camera within the object's intro radius
                    spawn = _instance_spawn_frame(rec, cam_samples) or 0
                try:
                    apply_instance_tracks(target, ar, scale, inst_name, spawn)
                    if spawn:
                        new_obj["dfx_spawn_frame"] = spawn
                        print(f"[DFX]     {inst_name}: track starts at intro frame "
                              f"{spawn + 1} (camera within {rec.intro_dist} units)")
                except Exception as e:
                    print(f"[DFX]   WARNING: track animation failed for {inst_name}: {e}")


def find_matching_vd3(dfx_path: str) -> Optional[str]:
    """Auto-detect VD3 file next to the DFX file."""
    directory = os.path.dirname(dfx_path)
    base = os.path.splitext(os.path.basename(dfx_path))[0]

    for ext in ('.VD3', '.vd3'):
        for name in (base, base.upper(), base.lower()):
            candidate = os.path.join(directory, name + ext)
            if os.path.isfile(candidate):
                return candidate

    # Scan directory
    try:
        for f in os.listdir(directory):
            if f.upper().endswith('.VD3'):
                if os.path.splitext(f)[0].upper() == base.upper():
                    return os.path.join(directory, f)
    except Exception:
        pass

    return None


# ===============================
#  PS1 VRAM / VRM texture system
# ===============================

@dataclass
class PS1Texture:
    """Represents one decoded PS1 texture page+CLUT combination."""
    index: int = 0
    tpage: int = 0
    clut: int = 0
    page_x: int = 0      # VRAM absolute X (pixel coords)
    page_y: int = 0       # VRAM absolute Y
    pmode: int = 0        # 0=4bit, 1=8bit, 2=16bit
    clut_x: int = 0       # CLUT absolute X
    clut_y: int = 0       # CLUT absolute Y
    width: int = 256      # texels wide (always 256 for 4bit/16bit, 128 for 8bit? No, 256 for all)
    height: int = 256     # texels tall
    tag: str = ""
    png_filename: str = ""


def decode_ps1_tpage(tpage):
    """Decode a PS1 tpage value into VRAM coordinates and pixel mode.
    Returns (page_x, page_y, pmode, abr).
    page_x/page_y are VRAM absolute pixel coordinates."""
    page_x = (tpage & 0x0F) * 64
    page_y = ((tpage >> 4) & 1) * 256
    abr = (tpage >> 5) & 3
    pmode = (tpage >> 7) & 3
    return page_x, page_y, pmode, abr


def decode_ps1_clut(clut_val):
    """Decode a PS1 CLUT value into VRAM coordinates.
    Returns (clut_x, clut_y) in VRAM absolute pixel coordinates."""
    clut_x = (clut_val & 0x3F) * 16
    clut_y = (clut_val >> 6) & 0x1FF
    return clut_x, clut_y


def load_vrm_vram(vrm_data):
    """Load VRM file as raw VRAM u16 array.
    VRM = 20-byte TIM-like header + 512*512 u16 pixels.
    Represents VRAM right half: x=512..1023, y=0..511."""
    if len(vrm_data) < 20 + 512 * 512 * 2:
        print(f"[PS1] WARNING: VRM file too small ({len(vrm_data)} bytes)")
        return None
    return vrm_data[20:20 + 512 * 512 * 2]


def _vrm_read_u16(vram_raw, vram_abs_x, vram_abs_y):
    """Read a u16 from VRM at absolute VRAM coordinates.
    VRM covers x=512..1023, so subtract 512 from X."""
    rx = vram_abs_x - 512
    if 0 <= rx < 512 and 0 <= vram_abs_y < 512:
        off = (vram_abs_y * 512 + rx) * 2
        if off + 2 <= len(vram_raw):
            return struct.unpack_from('<H', vram_raw, off)[0]
    return 0


def decode_ps1_texture_rgba(vram_raw, tpage, clut_val):
    """Decode a full 256x256 PS1 texture page to RGBA bytes.
    Uses the CLUT from the VRM for indexed modes.
    Returns (rgba_bytes, width, height)."""
    page_x, page_y, pmode, abr = decode_ps1_tpage(tpage)
    clut_x, clut_y = decode_ps1_clut(clut_val)

    # Read CLUT entries
    n_clut = 16 if pmode == 0 else 256 if pmode == 1 else 0
    clut = []
    for i in range(n_clut):
        clut.append(_vrm_read_u16(vram_raw, clut_x + i, clut_y))

    w, h = 256, 256
    rgba = bytearray(w * h * 4)

    for py in range(h):
        for px in range(w):
            if pmode == 0:  # 4-bit CLUT
                vram_x = page_x + (px // 4)
                nib_idx = px % 4
                raw = _vrm_read_u16(vram_raw, vram_x, page_y + py)
                idx = (raw >> (nib_idx * 4)) & 0xF
                color = clut[idx] if idx < len(clut) else 0
            elif pmode == 1:  # 8-bit CLUT
                vram_x = page_x + (px // 2)
                byte_idx = px % 2
                raw = _vrm_read_u16(vram_raw, vram_x, page_y + py)
                idx = (raw >> (byte_idx * 8)) & 0xFF
                color = clut[idx] if idx < len(clut) else 0
            else:  # 16-bit direct
                color = _vrm_read_u16(vram_raw, page_x + px, page_y + py)

            # Convert PS1 RGB555 to RGBA8888
            pi = (py * w + px) * 4
            if color == 0:
                rgba[pi:pi + 4] = b'\x00\x00\x00\x00'  # transparent
            else:
                r5 = color & 0x1F
                g5 = (color >> 5) & 0x1F
                b5 = (color >> 10) & 0x1F
                rgba[pi] = (r5 << 3) | (r5 >> 2)
                rgba[pi + 1] = (g5 << 3) | (g5 >> 2)
                rgba[pi + 2] = (b5 << 3) | (b5 >> 2)
                rgba[pi + 3] = 255

    return bytes(rgba), w, h


def save_ps1_texture_png(vram_raw, tpage, clut_val, out_path):
    """Decode a PS1 texture and save as PNG."""
    import zlib
    rgba, w, h = decode_ps1_texture_rgba(vram_raw, tpage, clut_val)

    def _make_chunk(chunk_type, chunk_data):
        c = chunk_type + chunk_data
        crc = struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack('>I', len(chunk_data)) + c + crc

    ihdr = struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)
    raw_rows = bytearray()
    for y in range(h):
        raw_rows.append(0)
        row_start = y * w * 4
        raw_rows.extend(rgba[row_start:row_start + w * 4])
    compressed = zlib.compress(bytes(raw_rows), 9)
    png = b'\x89PNG\r\n\x1a\n'
    png += _make_chunk(b'IHDR', ihdr)
    png += _make_chunk(b'IDAT', compressed)
    png += _make_chunk(b'IEND', b'')
    with open(out_path, 'wb') as f:
        f.write(png)


def collect_ps1_material_keys(data, is_level_polys=True, poly_start=0,
                               poly_count=0, mat_start=0):
    """Scan polygons and collect all unique (tpage, clut) combinations.
    Returns a set of (tpage, clut) tuples."""
    keys = set()
    for i in range(poly_count):
        off = poly_start + i * 12
        if off + 12 > len(data):
            break
        byte6 = data[off + 6]
        flags = data[off + 7]
        if byte6 == 0x05:
            continue
        if is_level_polys:
            mat_off = ru16(data, off + 10)
            if mat_off == 0xFFFF:
                continue
            maddr = mat_start + mat_off
        else:
            if not (flags & 0x02):
                continue
            maddr = ru32(data, off + 8)
        if maddr + 10 > len(data):
            continue
        clut = ru16(data, maddr + 2)
        tpage = ru16(data, maddr + 6)
        if clut != 0 or tpage != 0:
            keys.add((tpage, clut))
    return keys


def build_ps1_textures(vram_raw, material_keys, tex_dir):
    """Build PS1Texture list and extract PNG textures from VRM.
    material_keys: set of (tpage, clut) tuples.
    Returns (textures_list, key_to_index_dict)."""
    textures = []
    key_to_idx = {}

    os.makedirs(tex_dir, exist_ok=True)

    for tpage, clut_val in sorted(material_keys):
        idx = len(textures)
        page_x, page_y, pmode, abr = decode_ps1_tpage(tpage)
        clut_x, clut_y = decode_ps1_clut(clut_val)
        pm_names = ['4bit', '8bit', '16bit', '24bit']

        tex = PS1Texture()
        tex.index = idx
        tex.tpage = tpage
        tex.clut = clut_val
        tex.page_x = page_x
        tex.page_y = page_y
        tex.pmode = pmode
        tex.clut_x = clut_x
        tex.clut_y = clut_y
        tex.tag = f"tp{tpage:03X}_cl{clut_val:04X}_{pm_names[pmode]}"
        tex.png_filename = f"ps1_{idx:03d}_{tex.tag}.png"

        textures.append(tex)
        key_to_idx[(tpage, clut_val)] = idx

    return textures, key_to_idx


def load_ps1_textures_to_blender(vram_raw, textures, tex_dir):
    """Extract PS1 textures as PNG files and load them into Blender.
    Returns dict mapping texture index -> bpy.types.Image."""
    tex_images = {}
    for tex in textures:
        try:
            png_path = os.path.join(tex_dir, tex.png_filename)
            if not os.path.exists(png_path):
                save_ps1_texture_png(vram_raw, tex.tpage, tex.clut, png_path)
            img = bpy.data.images.load(png_path)
            img.alpha_mode = 'STRAIGHT'
            tex_images[tex.index] = img
        except Exception as e:
            print(f"[PS1] Warning: Failed to load texture {tex.index} '{tex.tag}': {e}")
    return tex_images


def _resolve_ps1_tpage(tpage, clut, key_to_idx):
    """Resolve a PS1 (tpage, clut) pair to a texture index."""
    key = (tpage, clut)
    if key in key_to_idx:
        return key_to_idx[key]
    # Try stripping blend mode bits from tpage
    stripped = tpage & ~0x60  # clear ABR bits
    key2 = (stripped, clut)
    if key2 in key_to_idx:
        return key_to_idx[key2]
    return -1


def find_matching_vrm(drm_path):
    """Auto-detect VRM file next to the DRM file."""
    directory = os.path.dirname(drm_path)
    base = os.path.splitext(os.path.basename(drm_path))[0]

    for ext in ('.vrm', '.VRM'):
        for name in (base, base.upper(), base.lower()):
            candidate = os.path.join(directory, name + ext)
            if os.path.isfile(candidate):
                return candidate
    try:
        for f in os.listdir(directory):
            if f.upper().endswith('.VRM'):
                if os.path.splitext(f)[0].upper() == base.upper():
                    return os.path.join(directory, f)
    except Exception:
        pass
    return None


def _ps1_get_or_create_material_textured(
    tpage, clut, ps1_textures, ps1_key_to_idx, tex_images, mat_cache
):
    """Get or create a Blender material for a PS1 textured polygon."""
    resolved = _resolve_ps1_tpage(tpage, clut, ps1_key_to_idx)
    if resolved < 0:
        return None

    cache_key = f"ps1_tex_{resolved:04d}"
    if cache_key in mat_cache:
        return mat_cache[cache_key]

    tex = ps1_textures[resolved]
    mat = bpy.data.materials.new(name=f"mat_{cache_key}_{_safe_name(tex.tag)}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for n in nodes:
        nodes.remove(n)

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    for spec_name in ('Specular IOR Level', 'Specular'):
        if spec_name in bsdf.inputs:
            bsdf.inputs[spec_name].default_value = 0.0
            break
    bsdf.inputs['Roughness'].default_value = 1.0

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    if resolved in tex_images:
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.location = (-400, 0)
        tex_node.image = tex_images[resolved]
        tex_node.interpolation = 'Closest'
        links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(tex_node.outputs['Alpha'], bsdf.inputs['Alpha'])
        try:
            mat.blend_method = 'CLIP'
        except Exception:
            pass
        try:
            mat.surface_render_method = 'DITHERED'
        except Exception:
            pass

    mat_cache[cache_key] = mat
    return mat


def create_blender_mesh_ps1(
    model, ps1_textures, ps1_key_to_idx, tex_images,
    mat_cache, scale, collection
):
    """Create a Blender mesh from a model using PS1 texture system."""
    if not model.vertices or not model.polygons:
        return None

    name = _safe_name(model.name)
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)

    # Gather materials
    mat_index_map = {}
    mat_list = []

    for poly in model.polygons:
        if poly.is_textured:
            resolved = _resolve_ps1_tpage(poly.tpage, poly.clut, ps1_key_to_idx)
            key = ('ps1tex', resolved if resolved >= 0 else -1)
        else:
            key = ('rgb', poly.color_r, poly.color_g, poly.color_b)

        if key not in mat_index_map:
            if key[0] == 'ps1tex' and key[1] >= 0:
                mat = _ps1_get_or_create_material_textured(
                    poly.tpage, poly.clut, ps1_textures,
                    ps1_key_to_idx, tex_images, mat_cache)
            elif key[0] == 'rgb':
                mat = _get_or_create_material_colored(
                    key[1], key[2], key[3], mat_cache)
            else:
                mat = _get_or_create_material_colored(128, 128, 128, mat_cache)

            if mat is not None:
                mat_index_map[key] = len(mat_list)
                mat_list.append(mat)
            else:
                mat_index_map[key] = 0

    for mat in mat_list:
        mesh.materials.append(mat)

    # Build geometry using BMesh
    verts_co = [(v.x * scale, v.y * scale, v.z * scale)
                for v in model.vertices]

    bm = bmesh.new()
    bm_verts = [bm.verts.new(co) for co in verts_co]
    bm.verts.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.new("UVMap")

    max_vi = len(model.vertices) - 1

    for poly in model.polygons:
        if not (0 <= poly.v1 <= max_vi and
                0 <= poly.v2 <= max_vi and
                0 <= poly.v3 <= max_vi and
                poly.v1 != poly.v2 and poly.v2 != poly.v3 and poly.v1 != poly.v3):
            continue

        try:
            face = bm.faces.new((bm_verts[poly.v1],
                                  bm_verts[poly.v2],
                                  bm_verts[poly.v3]))
        except ValueError:
            continue

        if poly.is_textured:
            resolved = _resolve_ps1_tpage(poly.tpage, poly.clut, ps1_key_to_idx)
            key = ('ps1tex', resolved if resolved >= 0 else -1)
        else:
            key = ('rgb', poly.color_r, poly.color_g, poly.color_b)

        if key in mat_index_map:
            face.material_index = mat_index_map[key]

        if poly.is_textured:
            if poly.is_overlay:
                vp1 = model.vertices[poly.v1]
                vp2 = model.vertices[poly.v2]
                vp3 = model.vertices[poly.v3]
                pts = [(vp1.x, vp1.y, vp1.z),
                       (vp2.x, vp2.y, vp2.z),
                       (vp3.x, vp3.y, vp3.z)]
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
                rx = max(xs) - min(xs); ry = max(ys) - min(ys); rz = max(zs) - min(zs)
                axes = sorted([(rx, 0), (ry, 1), (rz, 2)], reverse=True)
                a0 = axes[0][1]; a1 = axes[1][1]
                vals0 = [p[a0] for p in pts]; vals1 = [p[a1] for p in pts]
                r0 = max(vals0) - min(vals0); r1 = max(vals1) - min(vals1)
                if r0 < 1: r0 = 1
                if r1 < 1: r1 = 1
                for li, vp in enumerate(pts):
                    u = (vp[a0] - min(vals0)) / r0
                    v = (vp[a1] - min(vals1)) / r1
                    face.loops[li][uv_layer].uv = (u, v)
            else:
                face.loops[0][uv_layer].uv = (poly.u0, 1.0 - poly.v0)
                face.loops[1][uv_layer].uv = (poly.u1, 1.0 - poly.v1uv)
                face.loops[2][uv_layer].uv = (poly.u2, 1.0 - poly.v2uv)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    if model.metadata:
        for key, value in model.metadata.items():
            try:
                obj[f"dfx_{key}"] = value
            except Exception:
                obj[f"dfx_{key}"] = str(value)

    if model.bones and len(model.bones) > 1:
        try:
            _create_armature_for_mesh(obj, model, scale, collection)
        except Exception as e:
            print(f"[PS1] Warning: Failed to create armature for {name}: {e}")

    return obj


def load_drm(context, filepath, vrm_path="", import_level=True,
             import_objects=True, import_textures=True,
             import_overlays=True, import_animations=False, scale=0.01):
    """Import a PS1 DRM level file with VRM textures."""
    print(f"[PS1] Importing DRM: {filepath}")
    if not os.path.isfile(filepath):
        return {'CANCELLED'}

    base_name = os.path.splitext(os.path.basename(filepath))[0]

    if not vrm_path or not os.path.isfile(vrm_path):
        vrm_path = find_matching_vrm(filepath)

    # Read DRM — same bitshift as DFX
    with open(filepath, 'rb') as f:
        drm_raw = f.read()
    first_u32 = struct.unpack_from('<I', drm_raw, 0)[0]
    bitshift = ((first_u32 >> 9) << 11) + 0x800
    if bitshift >= len(drm_raw):
        print(f"[PS1] ERROR: Bitshift 0x{bitshift:X} exceeds file size")
        return {'CANCELLED'}
    data = drm_raw[bitshift:]
    print(f"[PS1] Data section: {len(data)} bytes (bitshift=0x{bitshift:X})")

    # -Load VRM and build PS1 textures
    vram_raw = None
    ps1_textures = []
    ps1_key_to_idx = {}
    tex_images = {}

    if vrm_path and os.path.isfile(vrm_path) and import_textures:
        print(f"[PS1] Loading VRM: {vrm_path}")
        with open(vrm_path, 'rb') as f:
            vrm_data = f.read()
        vram_raw = load_vrm_vram(vrm_data)

        if vram_raw:
            # Scan all materials in level geometry to collect texture keys
            lp = ru32(data, 0)
            all_keys = set()

            if import_level and lp + 0x48 <= len(data):
                ps_addr = ru32(data, lp + 0x38)
                pc = ru32(data, lp + 0x20)
                ms_addr = ru32(data, lp + 0x44)
                level_keys = collect_ps1_material_keys(
                    data, is_level_polys=True,
                    poly_start=ps_addr, poly_count=pc, mat_start=ms_addr)
                all_keys.update(level_keys)

            # Scan object models for texture keys
            if import_objects:
                try:
                    obj_models = extract_object_models(data)
                    for mdl, obj_addr in obj_models:
                        for poly in mdl.polygons:
                            if poly.is_textured and (poly.clut != 0 or poly.tpage != 0):
                                all_keys.add((poly.tpage, poly.clut))
                except Exception:
                    pass

            # Scan skybox materials
            try:
                segments = extract_level_anim_table(data)
                for seg in segments:
                    for tmat in seg.tri_materials:
                        if tmat is not None:
                            raw_tp = tmat[0]
                            # Skybox materials might use VD3-style tpage or PS1
                            # Check if it looks like a PS1 tpage
                            if raw_tp < 0x200:
                                # Could be PS1 tpage — get clut from the raw data
                                # For skybox, we need to re-read the material to get clut
                                pass
                            # We'll handle skybox separately below
            except Exception:
                pass

            print(f"[PS1] Found {len(all_keys)} unique (tpage, clut) combinations")

            tex_dir = os.path.join(os.path.dirname(filepath), f"{base_name}_ps1_textures")
            ps1_textures, ps1_key_to_idx = build_ps1_textures(
                vram_raw, all_keys, tex_dir)
            tex_images = load_ps1_textures_to_blender(
                vram_raw, ps1_textures, tex_dir)
            print(f"[PS1] Loaded {len(tex_images)} PS1 texture images into Blender")

    collection = bpy.data.collections.new(base_name)
    context.scene.collection.children.link(collection)
    mat_cache = {}

    # Level geometry
    overlay_list = []
    if import_level:
        try:
            level_model, overlay_list = extract_level_geometry(data)
            print(f"[PS1] Level geometry: {len(level_model.vertices)} verts, "
                  f"{len(level_model.polygons)} polys")
            level_obj = create_blender_mesh_ps1(
                level_model, ps1_textures, ps1_key_to_idx,
                tex_images, mat_cache, scale, collection)
            if level_obj:
                print(f"[PS1] Created level mesh: {level_obj.name}")
            if overlay_list:
                print(f"[PS1] Found {len(overlay_list)} overlay surfaces")
        except Exception as e:
            print(f"[PS1] ERROR extracting level geometry: {e}")
            import traceback; traceback.print_exc()

    # Object models
    obj_ptr_to_blender = {}
    obj_collection = None
    if import_objects:
        try:
            obj_models = extract_object_models(data)
            print(f"[PS1] Found {len(obj_models)} object models")
            obj_collection = bpy.data.collections.new(f"{base_name}_objects")
            collection.children.link(obj_collection)
            name_counter = {}
            for mdl, obj_addr in obj_models:
                if not mdl.vertices or not mdl.polygons:
                    continue
                if mdl.name in name_counter:
                    name_counter[mdl.name] += 1
                    mdl.name = f"{mdl.name}_{name_counter[mdl.name]:02d}"
                else:
                    name_counter[mdl.name] = 0
                try:
                    obj = create_blender_mesh_ps1(
                        mdl, ps1_textures, ps1_key_to_idx,
                        tex_images, mat_cache, scale, obj_collection)
                    if obj:
                        if obj_addr not in obj_ptr_to_blender:
                            obj_ptr_to_blender[obj_addr] = obj
                        print(f"[PS1]   Object: {obj.name} "
                              f"({len(mdl.vertices)}v, {len(mdl.polygons)}p)")
                        if obj.parent and obj.parent.type == 'ARMATURE':
                            arm_obj = obj.parent
                            bc = len(mdl.bones)
                            if bc > 1 and import_animations:
                                try:
                                    anims = parse_animations(data, obj_addr, bc)
                                    if anims:
                                        apply_animations_to_armature(arm_obj, anims, mdl.bones, scale)
                                except Exception as ae:
                                    print(f"[PS1]   WARNING: Animation import "
                                          f"failed for '{mdl.name}': {ae}")
                except Exception as e:
                    print(f"[PS1]   WARNING: Failed to create mesh for "
                          f"'{mdl.name}': {e}")
        except Exception as e:
            print(f"[PS1] ERROR extracting object models: {e}")
            import traceback; traceback.print_exc()

    # Placement table
    if import_objects:
        try:
            placements = extract_placement_table(data)
            if placements:
                placed_collection = bpy.data.collections.new(f"{base_name}_placed")
                collection.children.link(placed_collection)
                apply_placements(placements, obj_ptr_to_blender,
                                 scale, placed_collection)
                print(f"[PS1] Placed {len(placements)} instances")
            else:
                print(f"[PS1] No placement table found")
        except Exception as e:
            print(f"[PS1] WARNING: Failed to apply placements: {e}")
            import traceback; traceback.print_exc()

    # Reorder collections
    if obj_collection is not None:
        try:
            collection.children.unlink(obj_collection)
            collection.children.link(obj_collection)
        except Exception:
            pass
        try:
            vl = bpy.context.view_layer
            vl_col = vl.layer_collection.children[collection.name].children[
                obj_collection.name]
            vl_col.hide_viewport = True
        except Exception:
            pass

    # Overlay surfaces
    if import_overlays and overlay_list:
        try:
            overlay_collection = bpy.data.collections.new(f"{base_name}_overlay")
            collection.children.link(overlay_collection)
            for surf in overlay_list:
                sidx = surf['surface_index']
                mdl = ExtractedModel()
                mdl.name = f"overlay_{sidx:03d}"
                mdl.vertices = surf['vertices']
                mdl.polygons = surf['polys']
                obj = create_blender_mesh_ps1(
                    mdl, ps1_textures, ps1_key_to_idx,
                    tex_images, mat_cache, scale, overlay_collection)
                if obj:
                    obj["dfx_overlay"] = True
                    obj["dfx_surface_index"] = sidx
            print(f"[PS1] Created {len(overlay_list)} overlay surface meshes")
            try:
                vl = bpy.context.view_layer
                vl_col = vl.layer_collection.children[collection.name].children[
                    overlay_collection.name]
                vl_col.hide_viewport = True
            except Exception:
                pass
        except Exception as e:
            print(f"[PS1] WARNING: Failed to create overlay surfaces: {e}")
            import traceback; traceback.print_exc()

    # Skybox geometry
    try:
        segments = extract_level_anim_table(data)
        if segments:
            sky_collection = bpy.data.collections.new(f"{base_name}_skybox")
            collection.children.link(sky_collection)

            # For skybox, re-read materials to get PS1 clut values
            # create PS1 textured meshes
            if vram_raw and ps1_key_to_idx:
                _create_ps1_skybox_objects(
                    segments, scale, sky_collection, data,
                    ps1_textures, ps1_key_to_idx, tex_images,
                    vram_raw, mat_cache)
            else:
                create_level_anim_objects(
                    segments, scale, sky_collection, data)

            total_v = sum(len(s.vertices) for s in segments)
            total_t = sum(len(s.triangles) for s in segments)
            print(f"[PS1] Skybox: {len(segments)} elements, "
                  f"{total_v} vertices, {total_t} triangles")
        else:
            print(f"[PS1] No skybox data found")
    except Exception as e:
        print(f"[PS1] WARNING: Failed to parse skybox data: {e}")
        import traceback; traceback.print_exc()

    print(f"[PS1] Import complete. Materials: {len(mat_cache)}")
    return {'FINISHED'}


def _create_ps1_skybox_objects(segments, scale, collection, data,
                                ps1_textures, ps1_key_to_idx,
                                tex_images, vram_raw, mat_cache):
    """Create skybox mesh objects with PS1 textures.
    Re-reads triangle material pointers to extract CLUT values."""
    for si, seg in enumerate(segments):
        if not seg.vertices:
            continue

        mesh_name = f"sky_{si:02d}_t{seg.seg_type}"
        mesh = bpy.data.meshes.new(mesh_name)
        obj = bpy.data.objects.new(mesh_name, mesh)

        sky_scale = scale * 16.0
        verts_co = [(x * sky_scale, y * sky_scale, z * sky_scale)
                     for x, y, z in seg.vertices]

        faces = []
        if seg.triangles:
            for v0, v1, v2 in seg.triangles:
                if (v0 < len(verts_co) and v1 < len(verts_co) and
                        v2 < len(verts_co)):
                    faces.append((v0, v1, v2))

        if faces:
            mesh.from_pydata(verts_co, [], faces)
        else:
            mesh.from_pydata(verts_co, [], [])
        mesh.update()

        has_mats = (seg.tri_materials and len(seg.tri_materials) == len(faces))
        if has_mats:
            uv_layer = mesh.uv_layers.new(name="UVMap")
            tpage_to_mat_idx = {}

            for fi, tmat in enumerate(seg.tri_materials):
                if tmat is None:
                    continue
                raw_tpage = tmat[0]
                clut_val = tmat[7] if len(tmat) > 7 else 0

                # Collect any new (tpage, clut) combinations for skybox
                if clut_val != 0 and (raw_tpage, clut_val) not in ps1_key_to_idx:
                    # Dynamically add this texture
                    idx = len(ps1_textures)
                    page_x, page_y, pmode, abr = decode_ps1_tpage(raw_tpage)
                    clut_x, clut_y = decode_ps1_clut(clut_val)
                    pm_names = ['4bit', '8bit', '16bit', '24bit']
                    tex = PS1Texture()
                    tex.index = idx
                    tex.tpage = raw_tpage; tex.clut = clut_val
                    tex.page_x = page_x; tex.page_y = page_y; tex.pmode = pmode
                    tex.clut_x = clut_x; tex.clut_y = clut_y
                    tex.tag = f"tp{raw_tpage:03X}_cl{clut_val:04X}_{pm_names[pmode]}"
                    tex.png_filename = f"ps1_{idx:03d}_{tex.tag}.png"
                    ps1_textures.append(tex)
                    ps1_key_to_idx[(raw_tpage, clut_val)] = idx
                    # Extract texture
                    tex_dir = os.path.dirname(list(tex_images.values())[0].filepath) if tex_images else ""
                    if tex_dir and vram_raw:
                        try:
                            png_path = os.path.join(tex_dir, tex.png_filename)
                            if not os.path.exists(png_path):
                                save_ps1_texture_png(vram_raw, raw_tpage, clut_val, png_path)
                            img = bpy.data.images.load(png_path)
                            img.alpha_mode = 'STRAIGHT'
                            tex_images[idx] = img
                        except Exception:
                            pass

                resolved = _resolve_ps1_tpage(raw_tpage, clut_val, ps1_key_to_idx)
                if resolved < 0:
                    continue

                if resolved not in tpage_to_mat_idx:
                    mat = _ps1_get_or_create_material_textured(
                        raw_tpage, clut_val, ps1_textures,
                        ps1_key_to_idx, tex_images, mat_cache)
                    if mat:
                        obj.data.materials.append(mat)
                        tpage_to_mat_idx[resolved] = len(obj.data.materials) - 1

            for fi, face in enumerate(mesh.polygons):
                if fi >= len(seg.tri_materials):
                    break
                tmat = seg.tri_materials[fi]
                if tmat is None:
                    continue
                tpage = tmat[0]
                u0, v0, u1, v1, u2, v2 = tmat[1], tmat[2], tmat[3], tmat[4], tmat[5], tmat[6]
                clut_val = tmat[7] if len(tmat) > 7 else 0

                resolved = _resolve_ps1_tpage(tpage, clut_val, ps1_key_to_idx)
                if resolved in tpage_to_mat_idx:
                    face.material_index = tpage_to_mat_idx[resolved]

                for li, loop_idx in enumerate(face.loop_indices):
                    if li == 0:
                        uv_layer.data[loop_idx].uv = (u0, 1.0 - v0)
                    elif li == 1:
                        uv_layer.data[loop_idx].uv = (u1, 1.0 - v1)
                    elif li == 2:
                        uv_layer.data[loop_idx].uv = (u2, 1.0 - v2)

        obj["dfx_sky_type"] = seg.seg_type
        obj["dfx_sky_verts"] = len(seg.vertices)
        obj["dfx_sky_tris"] = len(faces)

        collection.objects.link(obj)
def load_dfx(context, filepath, vd3_path="", import_level=True,
             import_objects=True, import_textures=True,
             import_overlays=True, import_animations=False, scale=0.01):
    print(f"[DFX] Importing: {filepath}")
    if not os.path.isfile(filepath): return {'CANCELLED'}
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    if not vd3_path or not os.path.isfile(vd3_path):
        vd3_path = find_matching_vd3(filepath)
    with open(filepath, 'rb') as f: dfx_raw = f.read()
    first_u32 = struct.unpack_from('<I', dfx_raw, 0)[0]
    bitshift = ((first_u32 >> 9) << 11) + 0x800
    if bitshift >= len(dfx_raw):
        print(f"[DFX] ERROR: Bitshift 0x{bitshift:X} exceeds file size"); return {'CANCELLED'}
    data = dfx_raw[bitshift:]
    print(f"[DFX] Data section: {len(data)} bytes (bitshift=0x{bitshift:X})")

    textures = []; tex_images = {}; vd3_data = None
    if vd3_path and os.path.isfile(vd3_path) and import_textures:
        print(f"[DFX] Loading VD3: {vd3_path}")
        with open(vd3_path, 'rb') as f: vd3_data = f.read()
        textures = scan_vd3(vd3_data)
        print(f"[DFX] Found {len(textures)} textures in VD3")
        tex_dir = os.path.join(os.path.dirname(filepath), f"{base_name}_textures")
        tex_images = load_textures_to_blender(vd3_data, textures, tex_dir)
        print(f"[DFX] Loaded {len(tex_images)} texture images into Blender")

    collection = bpy.data.collections.new(base_name)
    context.scene.collection.children.link(collection)
    mat_cache = {}

    overlay_list = []
    if import_level:
        try:
            level_model, overlay_list = extract_level_geometry(data)
            print(f"[DFX] Level geometry: {len(level_model.vertices)} verts, {len(level_model.polygons)} polys")
            level_obj = create_blender_mesh(level_model, textures, tex_images, mat_cache, scale, collection)
            if level_obj: print(f"[DFX] Created level mesh: {level_obj.name}")
            if overlay_list:
                print(f"[DFX] Found {len(overlay_list)} overlay surfaces ({sum(len(o['polys']) for o in overlay_list)} polys)")
        except Exception as e:
            print(f"[DFX] ERROR extracting level geometry: {e}")
            import traceback; traceback.print_exc()

    obj_ptr_to_blender = {}
    obj_collection = None
    if import_objects:
        try:
            obj_models = extract_object_models(data)
            print(f"[DFX] Found {len(obj_models)} object models")
            obj_collection = bpy.data.collections.new(f"{base_name}_objects")
            # Link immediately so armature creation works (requires active object in linked collection)
            collection.children.link(obj_collection)
            name_counter = {}
            for mdl, obj_addr in obj_models:
                if not mdl.vertices or not mdl.polygons: continue
                if mdl.name in name_counter:
                    name_counter[mdl.name] += 1; mdl.name = f"{mdl.name}_{name_counter[mdl.name]:02d}"
                else: name_counter[mdl.name] = 0
                try:
                    obj = create_blender_mesh(mdl, textures, tex_images, mat_cache, scale, obj_collection)
                    if obj:
                        if obj_addr not in obj_ptr_to_blender: obj_ptr_to_blender[obj_addr] = obj
                        print(f"[DFX]   Object: {obj.name} ({len(mdl.vertices)}v, {len(mdl.polygons)}p)")
                        if obj.parent and obj.parent.type == 'ARMATURE':
                            arm_obj = obj.parent; bc = len(mdl.bones)
                            if bc > 1 and import_animations:
                                try:
                                    anims = parse_animations(data, obj_addr, bc)
                                    if anims: apply_animations_to_armature(arm_obj, anims, mdl.bones, scale)
                                except Exception as ae:
                                    print(f"[DFX]   WARNING: Animation import failed for '{mdl.name}': {ae}")
                except Exception as e:
                    print(f"[DFX]   WARNING: Failed to create mesh for '{mdl.name}': {e}")
        except Exception as e:
            print(f"[DFX] ERROR extracting object models: {e}")
            import traceback; traceback.print_exc()

    if import_objects:
        try:
            placements = extract_placement_table(data)
            if placements:
                placed_collection = bpy.data.collections.new(f"{base_name}_placed")
                collection.children.link(placed_collection)  # linked FIRST
                apply_placements(placements, obj_ptr_to_blender, scale, placed_collection)
                print(f"[DFX] Placed {len(placements)} instances")
            else: print(f"[DFX] No placement table found")
        except Exception as e:
            print(f"[DFX] WARNING: Failed to apply placements: {e}")
            import traceback; traceback.print_exc()

    # Reorder collections: unlink _objects and relink after _placed for outliner order
    if obj_collection is not None:
        try:
            collection.children.unlink(obj_collection)
            collection.children.link(obj_collection)
        except Exception:
            pass
        # Hide the collection (not individual objects) in viewport
        try:
            vl = bpy.context.view_layer
            vl_col = vl.layer_collection.children[collection.name].children[obj_collection.name]
            vl_col.hide_viewport = True
        except Exception:
            pass

    # Overlay surfaces as individual meshes
    if import_overlays and overlay_list:
        try:
            overlay_collection = bpy.data.collections.new(f"{base_name}_overlay")
            collection.children.link(overlay_collection)
            for surf in overlay_list:
                sidx = surf['surface_index']
                mdl = ExtractedModel()
                mdl.name = f"overlay_{sidx:03d}"
                mdl.vertices = surf['vertices']
                mdl.polygons = surf['polys']
                obj = create_blender_mesh(mdl, textures, tex_images, mat_cache, scale, overlay_collection)
                if obj:
                    # Store all raw data as custom properties for reverse engineering
                    obj["dfx_overlay"] = True
                    obj["dfx_surface_index"] = sidx
                    obj["dfx_effect_type"] = surf.get('effect_type_index', -1)
                    obj["dfx_poly_count"] = len(surf['polys'])
                    obj["dfx_val8_list"] = str(surf['val8_list'])
                    obj["dfx_flags_list"] = str([f"0x{f:02X}" for f in surf['flags_list']])
                    obj["dfx_poly_raw"] = str(surf['poly_raw'])
                    obj["dfx_overlay_count"] = surf['overlay_count']

                    # Effect type table summary
                    if surf['effect_types']:
                        et_summary = []
                        for et in surf['effect_types']:
                            tname = textures[et['base_tex']].tag if et['base_tex'] < len(textures) else str(et['base_tex'])
                            et_summary.append(f"[{et['index']}] tex={et['base_tex']}({tname}) {et['width']}x{et['height']} anim=({et['anim_type']},{et['frame_count']})")
                        obj["dfx_effect_types"] = str(et_summary)

                        # Frame lists per effect
                        for et in surf['effect_types']:
                            if et['frames']:
                                frame_strs = []
                                for fr in et['frames']:
                                    fn = textures[fr['tex']].tag if fr['tex'] < len(textures) else str(fr['tex'])
                                    frame_strs.append(f"{fr['tex']}({fn}) p={fr['param']}")
                                obj[f"dfx_effect{et['index']}_frames"] = str(frame_strs)

                    # Overlay table entry for this surface
                    ote = surf['overlay_table_entry']
                    if ote:
                        obj["dfx_ovl_entry_v0"] = ote['v0']
                        obj["dfx_ovl_entry_v1"] = ote['v1']
                        if 'vfx_ptr' in ote:
                            obj["dfx_ovl_vfx_ptr"] = f"0x{ote['vfx_ptr']:X}"
                            obj["dfx_ovl_vfx_raw"] = ote['vfx_raw']

            print(f"[DFX] Created {len(overlay_list)} overlay surface meshes")
            # Hide overlay collection in viewport
            try:
                vl = bpy.context.view_layer
                vl_col = vl.layer_collection.children[collection.name].children[overlay_collection.name]
                vl_col.hide_viewport = True
            except Exception:
                pass
        except Exception as e:
            print(f"[DFX] WARNING: Failed to create overlay surfaces: {e}")
            import traceback; traceback.print_exc()

    # Skybox geometry (data section +0x20/+0x24)
    try:
        segments = extract_level_anim_table(data)
        if segments:
            sky_collection = bpy.data.collections.new(f"{base_name}_skybox")
            collection.children.link(sky_collection)
            sky_objs = create_level_anim_objects(
                segments, scale, sky_collection, data,
                textures=textures, tex_images=tex_images, mat_cache=mat_cache)
            total_v = sum(len(s.vertices) for s in segments)
            total_t = sum(len(s.triangles) for s in segments)
            print(f"[DFX] Skybox: {len(segments)} elements, "
                  f"{total_v} vertices, {total_t} triangles")
        else:
            print(f"[DFX] No skybox data found")
    except Exception as e:
        print(f"[DFX] WARNING: Failed to parse skybox data: {e}")
        import traceback; traceback.print_exc()

    print(f"[DFX] Import complete. Materials: {len(mat_cache)}")
    return {'FINISHED'}
