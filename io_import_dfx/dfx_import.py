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
    bone_index: int = 0; axis: int = 0; component_type: int = 0
    component: str = 'R'  # 'R'=rotation, 'T'=translation, 'S'=scale
    keyframes: List[float] = field(default_factory=list)

@dataclass
class ParsedAnimation:
    name: str = ""; frame_count: int = 0; w1: int = 0; ch: int = 0
    is_raw: bool = False; channels: List[AnimChannel] = field(default_factory=list)

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
#  Animation parsing (UNFINISHED and non-functional, removed from operator)
# =========================================================================

_CH_TO_BPA = {1: 1, 5: 2, 7: 3}

def _gte_to_radians(gte_val):
    """Convert a GTE fixed-point rotation value to radians.
    GTE convention: 4096 = one full revolution (360°).
    Value should already be phase-unwrapped before calling this."""
    import math
    return (gte_val / 4096.0) * 2.0 * math.pi

def _unwrap_gte_channel(values):
    """Phase-unwrap a channel of GTE rotation values.
    The GTE engine wraps internally (mod 4096), so accumulated values like
    16409 actually mean 25 (= 2.2°). We unwrap to keep values smooth
    and in the correct range.
    Returns new list of unwrapped values."""
    if not values:
        return values
    result = []
    # Wrap first value to [-2048, 2047]
    v = values[0] % 4096
    if v >= 2048: v -= 4096
    result.append(v)
    for i in range(1, len(values)):
        v = values[i] % 4096
        if v >= 2048: v -= 4096
        # Maintain continuity: adjust by 4096 to minimize jump from previous
        prev = result[-1]
        while v - prev > 2048: v -= 4096
        while v - prev < -2048: v += 4096
        result.append(v)
    return result

def _decode_anim_bitmask(data, p_hdr, bpa, bone_count):
    """Decode the animation bitmask into a list of channels.
    Each non-zero bitmask entry = exactly 1 curve slot.
    Bitmask is bone-major with 3 bits per bone.
    Bit-to-axis mapping: sequential (bit0→axis0, bit1→axis1, bit2→axis2).
    Returns (channels, n_curve_slots)."""
    bitmask_start = p_hdr + 2
    channels = []
    bit_pos = 0
    total_bits = bone_count * 3 * bpa
    for bone in range(bone_count):
        for axis in range(3):
            if bit_pos >= total_bits: break
            bval = 0
            for b in range(bpa):
                byte_i = (bit_pos + b) // 8
                bit_i = (bit_pos + b) % 8
                if bitmask_start + byte_i < len(data):
                    bval |= ((data[bitmask_start + byte_i] >> bit_i) & 1) << b
            if bval > 0:
                c = AnimChannel()
                c.bone_index = bone
                c.axis = axis
                c.component_type = bval
                c.component = 'R'
                channels.append(c)
            bit_pos += bpa
    return channels, len(channels)

def _try_raw_i16_decode(data, p_crv, crv_bytes, n_slots, frame_count):
    """Try to decode curve data as sequential raw i16 per channel slot.
    n_slots is the number of value slots from the expanded bitmask.
    Only succeeds if the data size is plausible for raw storage."""
    if n_slots == 0 or frame_count < 1:
        return None
    
    # Quick reject: if data is less than 50% of expected raw size, it's compressed
    expected_raw = n_slots * frame_count * 2
    if crv_bytes < expected_raw * 0.5:
        return None
    
    total_i16 = crv_bytes // 2
    
    # Find best sample count
    candidates = set()
    if total_i16 >= n_slots:
        exact = total_i16 // n_slots
        for delta in range(-2, 3):
            c = exact + delta
            if c >= frame_count - 2 and c <= frame_count + 5:
                if n_slots * c * 2 <= crv_bytes + 4:
                    candidates.add(c)
    if n_slots * frame_count * 2 <= crv_bytes + 4:
        candidates.add(frame_count)
    if n_slots * (frame_count + 1) * 2 <= crv_bytes + 4:
        candidates.add(frame_count + 1)
    
    if not candidates:
        return None
    
    best_ns = None
    best_smooth = -1
    for ns in sorted(candidates):
        if n_slots * ns * 2 > crv_bytes + 4:
            continue
        if ns < max(3, frame_count - 5):
            continue
        if p_crv + n_slots * ns * 2 > len(data):
            continue
        
        smooth = 0
        for ci in range(n_slots):
            off = p_crv + ci * ns * 2
            if off + ns * 2 > len(data):
                break
            vals = [ri16(data, off + i * 2) for i in range(ns)]
            max_d = 0
            for i in range(1, ns):
                d = abs(vals[i] - vals[i - 1])
                if d > 2048:
                    d = 4096 - d
                max_d = max(max_d, d)
            if max_d < 400:
                smooth += 1
        if smooth > best_smooth:
            best_smooth = smooth
            best_ns = ns
    
    if best_ns is None or best_smooth < max(1, int(n_slots * 0.6)):
        return None
    
    # Decode: return list of lists (one per slot)
    result = []
    for ci in range(n_slots):
        off = p_crv + ci * best_ns * 2
        if off + best_ns * 2 > len(data):
            result.append([0.0] * frame_count)
            continue
        raw = [ri16(data, off + i * 2) for i in range(best_ns)]
        if len(raw) >= frame_count:
            result.append([float(raw[i]) for i in range(frame_count)])
        else:
            out = []
            for fi in range(frame_count):
                t = fi * (len(raw) - 1) / max(1, frame_count - 1)
                idx = int(t); frac = t - idx
                if idx >= len(raw) - 1:
                    out.append(float(raw[-1]))
                else:
                    out.append(raw[idx] * (1.0 - frac) + raw[idx + 1] * frac)
            result.append(out)
    return result


def _decode_compressed_blocks(data, p_crv, crv_bytes, n_slots, frame_count):
    """Decode block-compressed curve data.
    Block stream format: [lo_byte, mode_byte, data...]
      mode 0x60: constant – (lo-1) i16 values, each a channel held for all frames
      mode 0x40: 4-bit nibble delta – channels with i16 init + packed nibble deltas
      mode 0x20: 8-bit byte delta – channels with i16 init + packed byte deltas
    Returns list of lists (one per decoded slot), or None on failure."""
    if crv_bytes < 2 or frame_count < 1:
        return None
    
    end = min(p_crv + crv_bytes, len(data))
    result = []
    pos = p_crv
    
    nib_bytes_per_ch = ((frame_count - 1) + 1) // 2  # ceil(deltas/2)
    byte_per_ch = frame_count - 1
    
    while pos + 2 <= end and len(result) < n_slots + 50:
        lo = data[pos]
        mode = data[pos + 1]
        
        if mode not in (0x20, 0x40, 0x60) or lo < 2:
            break
        
        block_bytes = lo * 2
        if pos + block_bytes > end:
            break
        
        data_words = lo - 1  # words of data after header
        
        if mode == 0x60:
            # Constant: each data word is one channel's value for all frames
            for wi in range(data_words):
                v = ri16(data, pos + 2 + wi * 2)
                result.append([float(v)] * frame_count)
        
        elif mode == 0x40:
            # 4-bit nibble delta channels
            # Per channel: 2-byte i16 init + nib_bytes_per_ch packed nibble bytes
            bpc = 2 + nib_bytes_per_ch
            n_ch = (data_words * 2) // bpc  # how many channels fit
            for ci in range(n_ch):
                ch_off = pos + 2 + ci * bpc
                if ch_off + bpc > end:
                    break
                init = ri16(data, ch_off)
                nibbles = []
                for nb in range(nib_bytes_per_ch):
                    bv = data[ch_off + 2 + nb]
                    nibbles.append(bv & 0x0F)
                    nibbles.append((bv >> 4) & 0x0F)
                # Convert to signed 4-bit: 8-15 → negative
                signed = [(n - 16) if n >= 8 else n for n in nibbles[:frame_count - 1]]
                vals = [float(init)]
                for d in signed:
                    vals.append(vals[-1] + d)
                # Pad to frame_count if needed
                while len(vals) < frame_count:
                    vals.append(vals[-1])
                result.append(vals[:frame_count])
        
        elif mode == 0x20:
            # 8-bit byte delta channels
            # Per channel: 2-byte i16 init + (frame_count-1) byte deltas
            bpc = 2 + byte_per_ch
            n_ch = (data_words * 2) // bpc
            for ci in range(n_ch):
                ch_off = pos + 2 + ci * bpc
                if ch_off + bpc > end:
                    break
                init = ri16(data, ch_off)
                vals = [float(init)]
                for d in range(byte_per_ch):
                    b = data[ch_off + 2 + d]
                    delta = b - 256 if b >= 128 else b
                    vals.append(vals[-1] + delta)
                while len(vals) < frame_count:
                    vals.append(vals[-1])
                result.append(vals[:frame_count])
        
        pos += block_bytes
    
    if len(result) == 0:
        return None
    return result

def parse_animations(data, obj_addr, bone_count):
    val14 = ru32(data, obj_addr + 0x14)
    if val14 == 0 or val14 >= len(data): return []
    anim_ptrs = []
    for j in range(0, 80, 4):
        if val14 + j + 4 > len(data): break
        v = ru32(data, val14 + j)
        if 0x1000 < v < len(data): anim_ptrs.append(v)
        else: break
    if not anim_ptrs: return []
    results = []
    for ai, aptr in enumerate(anim_ptrs):
        if aptr + 16 > len(data): continue
        frame_count = ru16(data, aptr); w1 = ru16(data, aptr + 2)
        p_hdr = ru32(data, aptr + 4); p_crv = ru32(data, aptr + 8)
        p_name = ru32(data, aptr + 12)
        if not (0 < p_hdr < len(data) and 0 < p_crv < len(data)): continue
        if frame_count == 0 or frame_count > 2000 or p_crv <= p_hdr: continue
        anim_name = f"anim_{ai}"
        if 0x1000 < p_name < len(data) - 12:
            raw_n = data[p_name:p_name + 12]; end = raw_n.find(0)
            if end > 0:
                try: anim_name = raw_n[:end].decode('ascii')
                except: pass
        ch_byte = data[p_hdr]
        if ch_byte == 0: continue
        bpa = _CH_TO_BPA.get(ch_byte, bin(ch_byte).count('1'))
        if bpa == 0: continue
        hdr_size = p_crv - p_hdr; bitmask_bytes = hdr_size - 2
        if bitmask_bytes <= 0: continue
        channels, n_slots = _decode_anim_bitmask(data, p_hdr, bpa, bone_count)
        if not channels: continue
        if ai + 1 < len(anim_ptrs): crv_end = anim_ptrs[ai + 1]
        else: crv_end = p_crv + min(n_slots * frame_count * 2 + 512, len(data) - p_crv)
        crv_end = min(crv_end, len(data)); crv_bytes = max(0, crv_end - p_crv)
        
        anim = ParsedAnimation(); anim.name = anim_name
        anim.frame_count = frame_count; anim.w1 = w1; anim.ch = ch_byte
        anim.channels = channels
        
        # Try raw i16 decode first
        raw_result = _try_raw_i16_decode(data, p_crv, crv_bytes, n_slots, frame_count)
        if raw_result is not None and len(raw_result) >= len(channels):
            anim.is_raw = True
            for ci, ch_obj in enumerate(channels):
                if ci < len(raw_result):
                    ch_obj.keyframes = _unwrap_gte_channel(raw_result[ci])
                else:
                    ch_obj.keyframes = [0.0] * frame_count
            print(f"[DFX]     Animation '{anim_name}' raw ({len(channels)}ch/{n_slots}slots, {frame_count}f)")
        else:
            # Try compressed block decode
            block_result = _decode_compressed_blocks(data, p_crv, crv_bytes, n_slots, frame_count)
            if block_result is not None and len(block_result) > 0:
                anim.is_raw = False
                # Map decoded slots to channels in order
                for ci, ch_obj in enumerate(channels):
                    if ci < len(block_result):
                        ch_obj.keyframes = _unwrap_gte_channel(block_result[ci])
                    else:
                        ch_obj.keyframes = [0.0] * frame_count
                decoded_count = min(len(block_result), len(channels))
                print(f"[DFX]     Animation '{anim_name}' compressed: decoded {decoded_count}/{len(channels)}ch ({len(block_result)} slots from blocks)")
            else:
                anim.is_raw = False
                for c in channels: c.keyframes = [0.0] * frame_count
                print(f"[DFX]     Animation '{anim_name}' UNDECODED (ch=0x{ch_byte:02X}, {len(channels)}ch, {crv_bytes}B)")
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


def save_texture_png(data: bytes, tex: VD3Texture, out_path: str):
    """Save a VD3 texture as a PNG file using pure Python (minimal dependencies)."""
    import zlib

    w, h = tex.width, tex.height
    rgba = decode_vd3_texture_rgba(data, tex)

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

        tail_offset = (0, 0, 10 * scale)  # default: 10 units up

        for ci, cb in enumerate(bones):
            if cb.parent_id == i and ci != i:
                cx, cy, cz = bone_world_pos[ci]
                dx = (cx - hx) * scale
                dy = (cy - hy) * scale
                dz = (cz - hz) * scale
                length = (dx*dx + dy*dy + dz*dz) ** 0.5
                if length > 0.001:
                    tail_offset = (dx, dy, dz)
                break

        eb.tail = (eb.head[0] + tail_offset[0],
                   eb.head[1] + tail_offset[1],
                   eb.head[2] + tail_offset[2])

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




def apply_animations_to_armature(arm_obj, animations):
    import math
    if not animations or arm_obj.type != 'ARMATURE': return
    for pb in arm_obj.pose.bones: pb.rotation_mode = 'ZYX'
    if arm_obj.animation_data is None: arm_obj.animation_data_create()
    
    # Detect Blender API: 4.4+ removed action.fcurves in favor of slotted actions
    _use_legacy_fcurves = True
    try:
        test_action = bpy.data.actions.new(name="__dfx_api_test__")
        _ = test_action.fcurves
        bpy.data.actions.remove(test_action)
    except AttributeError:
        _use_legacy_fcurves = False
        try: bpy.data.actions.remove(test_action)
        except: pass
    
    created_actions = []
    for anim in animations:
        if not anim.channels: continue
        has_data = any(
            len(c.keyframes) > 0 and any(abs(v) > 0.01 for v in c.keyframes)
            for c in anim.channels
        )
        if not has_data:
            print(f"[DFX]     SKIP '{anim.name}': all keyframes zero")
            continue
        # Diagnostics
        anim_ch = 0; const_ch = 0; zero_ch = 0
        for c in anim.channels:
            if not c.keyframes or len(c.keyframes) == 0: zero_ch += 1
            elif all(abs(v - c.keyframes[0]) < 0.01 for v in c.keyframes): const_ch += 1
            else: anim_ch += 1
        all_rot_vals = []
        for c in anim.channels:
            if c.keyframes:
                all_rot_vals.extend(c.keyframes)
        rot_range = ""
        if all_rot_vals:
            mn_raw = min(all_rot_vals); mx_raw = max(all_rot_vals)
            mn_deg = mn_raw * 360.0 / 4096.0
            mx_deg = mx_raw * 360.0 / 4096.0
            rot_range = f" deg=[{mn_deg:.1f},{mx_deg:.1f}]"
        print(f"[DFX]     '{anim.name}': {anim_ch} animated + {const_ch} const + {zero_ch} zero = {len(anim.channels)} ch, {anim.frame_count}f{rot_range}")
        
        action_name = f"{arm_obj.name}_{anim.name}"
        action = bpy.data.actions.new(name=action_name)
        action.use_fake_user = True
        action["dfx_frame_count"] = anim.frame_count
        action["dfx_w1"] = anim.w1; action["dfx_ch"] = anim.ch
        action["dfx_is_raw"] = anim.is_raw
        action["dfx_num_channels"] = len(anim.channels)
        action["dfx_animated_channels"] = anim_ch
        
        has_fcurves = False
        bones_missing = set()
        
        if _use_legacy_fcurves:
            # Legacy path: directly create fcurves on the action (Blender <4.4)
            for channel in anim.channels:
                bone_name = f"bone_{channel.bone_index:02d}"
                pb = arm_obj.pose.bones.get(bone_name)
                if pb is None: bones_missing.add(bone_name); continue
                if not channel.keyframes: continue
                data_path = f'pose.bones["{bone_name}"].rotation_euler'
                fcurve = action.fcurves.new(data_path=data_path, index=channel.axis)
                kf_points = fcurve.keyframe_points
                kf_points.add(len(channel.keyframes))
                for fi, gte_val in enumerate(channel.keyframes):
                    kf_points[fi].co = (float(fi + 1), _gte_to_radians(gte_val))
                    kf_points[fi].interpolation = 'LINEAR'
                fcurve.update()
                has_fcurves = True
        else:
            # Modern path: use keyframe_insert (Blender 4.4+)
            arm_obj.animation_data.action = action
            for fi in range(anim.frame_count):
                bpy.context.scene.frame_set(fi + 1)
                for channel in anim.channels:
                    bone_name = f"bone_{channel.bone_index:02d}"
                    pb = arm_obj.pose.bones.get(bone_name)
                    if pb is None:
                        if fi == 0: bones_missing.add(bone_name)
                        continue
                    if not channel.keyframes or fi >= len(channel.keyframes): continue
                    val = _gte_to_radians(channel.keyframes[fi])
                    pb.rotation_euler[channel.axis] = val
                    pb.keyframe_insert(data_path="rotation_euler", index=channel.axis, frame=fi + 1)
                    has_fcurves = True
            # Set interpolation to LINEAR for all created fcurves
            try:
                for fc in action.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = 'LINEAR'
            except: pass
        
        if bones_missing:
            print(f"[DFX]       WARNING: {len(bones_missing)} bones not found: {sorted(bones_missing)[:5]}...")
        if has_fcurves:
            created_actions.append(action)
        else:
            bpy.data.actions.remove(action)
            print(f"[DFX]       WARNING: no fcurves created for '{anim.name}'")
    
    if created_actions:
        arm_obj.animation_data.action = created_actions[0]
        for action in created_actions[1:]:
            track = arm_obj.animation_data.nla_tracks.new()
            track.name = action.name
            strip = track.strips.new(action.name, int(1), action)
            strip.action_frame_end = float(action.get("dfx_frame_count", 30))
            track.mute = True
    print(f"[DFX] Applied {len(created_actions)} actions to {arm_obj.name}")


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
        anim_ptr = ru32(data, roff + 0x24)
        if anim_ptr > 0 and anim_ptr < len(data):
            rec.anim_rec = parse_instance_anim_record(data, anim_ptr)
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



def parse_instance_anim_record(data, rec_addr):
    """Parse a per-instance animation record at the given address.
    Returns a dict with record type, size, tilt angle, and path waypoints."""
    import math
    if rec_addr <= 0 or rec_addr + 0x40 > len(data):
        return None
    
    # Find record size by scanning for CDCD terminator
    rec_size = -1
    for scan in range(0, 8000, 2):
        if rec_addr + scan + 2 > len(data):
            break
        if ru16(data, rec_addr + scan) == 0xCDCD:
            rec_size = scan + 2
            break
    
    flags38 = ru32(data, rec_addr + 0x38) if rec_addr + 0x3C <= len(data) else 0
    p20 = ru32(data, rec_addr + 0x20) if rec_addr + 0x24 <= len(data) else 0
    p34 = ru32(data, rec_addr + 0x34) if rec_addr + 0x38 <= len(data) else 0
    
    result = {
        'addr': rec_addr,
        'size': rec_size,
        'flags': flags38,
        'waypoints': [],
    }
    
    # simple_coord (112 bytes): static tilt matrix
    if rec_size == 112 and flags38 in (0x04010005,):
        result['type'] = 'simple_coord'
        A = ri16(data, rec_addr + 0x42)
        B = ri16(data, rec_addr + 0x44)
        if A != 0 or B != 0:
            result['tilt_angle_rad'] = math.atan2(-B, A)
        else:
            result['tilt_angle_rad'] = 0.0
        return result
    
    # Type B: animated_path (data at +0x34, flags at +0x38)
    # Used by airbal, tttruck (0x0400), msfairy, scripted entities (0x0100)
    # Frame-by-frame positions in 40-byte blocks
    # Same field layout as large_path: +0x02=X, +0x04=Y, +0x06=Z (all i16)
    f38_hi = flags38 >> 16
    if p34 > 0 and p34 < len(data) and f38_hi in (0x0400, 0x0100):
        result['type'] = 'animated_path'
        frame_count = flags38 & 0xFFFF
        waypoints = []
        for fi in range(frame_count):
            foff = p34 + fi * 40
            if foff + 8 > len(data):
                break
            x = ri16(data, foff + 2)        # i16 position X
            y = ri16(data, foff + 4)        # i16 position Y
            z = ri16(data, foff + 6)        # i16 position Z (height)
            waypoints.append((float(x), float(y), float(z)))
        result['waypoints'] = waypoints
        return result
    
    # Type A: large_path / AI path (data at +0x20)
    # 40-byte waypoint blocks at rec+0x50
    # Position: i16 X at +0x02, i16 Y at +0x04, i16 Z at +0x06
    # Read until seg=0 terminator (u16 at +0x00 of block)
    if p20 > 0 and p20 < len(data):
        result['type'] = 'large_path'
        
        waypoints = []
        for wi in range(2000):  # safety limit
            base = rec_addr + 0x50 + wi * 40
            if base + 8 > len(data):
                break
            seg = ru16(data, base)
            if seg == 0 and wi > 0:
                break  # terminator
            x = ri16(data, base + 2)        # i16 position X
            y = ri16(data, base + 4)        # i16 position Y
            z = ri16(data, base + 6)        # i16 position Z (height)
            waypoints.append((float(x), float(y), float(z)))
        result['waypoints'] = waypoints
        return result
    
    # Unknown record type
    result['type'] = 'unknown'
    return result


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

    for rec in placements:
        # Handle null-model instances (camera paths, sound emitters, triggers)
        if rec.obj_ptr not in obj_ptr_to_blender:
            if rec.obj_ptr == 0 and hasattr(rec, 'anim_rec') and rec.anim_rec is not None:
                # Create an Empty for invisible entities with animation
                ar = rec.anim_rec
                anim_type = ar.get('type', 'none')
                wps = ar.get('waypoints', [])
                if anim_type in ('animated_path', 'large_path') and len(wps) >= 2:
                    path_name = f"camera_path_{len(wps)}f"
                    curve_data = bpy.data.curves.new(path_name, type='CURVE')
                    curve_data.dimensions = '3D'
                    curve_data.resolution_u = 12
                    spline = curve_data.splines.new('POLY')
                    spline.points.add(len(wps) - 1)
                    for wi, (wx, wy, wz) in enumerate(wps):
                        spline.points[wi].co = (
                            wx * scale, wy * scale, wz * scale, 1.0)
                    path_obj = bpy.data.objects.new(path_name, curve_data)
                    path_obj["dfx_path_type"] = anim_type
                    path_obj["dfx_path_points"] = len(wps)
                    path_obj["dfx_null_model"] = True
                    path_obj.display_type = 'WIRE'
                    path_obj.show_in_front = True
                    collection.objects.link(path_obj)
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
            new_arm.rotation_euler = (0, 0, rot_rad)
            collection.objects.link(new_arm)

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
            new_obj.rotation_euler = (0, 0, rot_rad)
            collection.objects.link(new_obj)

        # Store placement metadata
        new_obj["dfx_placement"] = True
        new_obj["dfx_rotation_deg"] = round((rec.rotation_raw / 4096.0) * 360.0, 1)
        new_obj["dfx_rotation_raw"] = rec.rotation_raw
        new_obj["dfx_flags"] = rec.flags
        new_obj["dfx_draw_distance"] = rec.draw_distance

        # Parse per-instance animation record if present
        if hasattr(rec, 'anim_rec') and rec.anim_rec is not None:
            ar = rec.anim_rec
            anim_type = ar.get('type', 'none')
            new_obj["dfx_anim_type"] = anim_type
            new_obj["dfx_anim_size"] = ar.get('size', -1)
            
            if anim_type == 'simple_coord':
                tilt_rad = ar.get('tilt_angle_rad', 0.0)
                new_obj["dfx_tilt_angle_deg"] = round(
                    tilt_rad * 180.0 / 3.14159265, 2)
                # Apply tilt as X-axis rotation on top of Y heading
                target = new_arm if source_arm else new_obj
                target.rotation_euler = (tilt_rad,
                                         target.rotation_euler[1],
                                         target.rotation_euler[2])
            
            elif anim_type == 'animated_path':
                wps = ar.get('waypoints', [])
                if len(wps) >= 2:
                    path_name = f"{inst_name}_path"
                    curve_data = bpy.data.curves.new(path_name, type='CURVE')
                    curve_data.dimensions = '3D'
                    curve_data.resolution_u = 12
                    spline = curve_data.splines.new('POLY')
                    spline.points.add(len(wps) - 1)
                    for wi, (wx, wy, wz) in enumerate(wps):
                        spline.points[wi].co = (
                            wx * scale, wy * scale, wz * scale, 1.0)
                    path_obj = bpy.data.objects.new(path_name, curve_data)
                    path_obj["dfx_path_type"] = anim_type
                    path_obj["dfx_path_points"] = len(wps)
                    path_obj.display_type = 'WIRE'
                    path_obj.show_in_front = True
                    collection.objects.link(path_obj)
                    new_obj["dfx_path_waypoints"] = len(wps)
            
            elif anim_type == 'large_path':
                # AI path — absolute coordinates, create cyclic poly curve
                wps = ar.get('waypoints', [])
                if len(wps) >= 2:
                    path_name = f"{inst_name}_path"
                    curve_data = bpy.data.curves.new(path_name, type='CURVE')
                    curve_data.dimensions = '3D'
                    curve_data.resolution_u = 12
                    spline = curve_data.splines.new('POLY')
                    spline.points.add(len(wps) - 1)
                    for wi, (wx, wy, wz) in enumerate(wps):
                        spline.points[wi].co = (
                            wx * scale, wy * scale, wz * scale, 1.0)
                    path_obj = bpy.data.objects.new(path_name, curve_data)
                    path_obj["dfx_path_type"] = anim_type
                    path_obj["dfx_path_points"] = len(wps)
                    path_obj.display_type = 'WIRE'
                    path_obj.show_in_front = True
                    collection.objects.link(path_obj)




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
                                        apply_animations_to_armature(arm_obj, anims)
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
                                    if anims: apply_animations_to_armature(arm_obj, anims)
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
