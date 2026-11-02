#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

// Parse baseline sensor JPEG for RFC 2435 RTP/JPEG. Header-only for host tests.
struct JpegScan {
  const uint8_t* scan = nullptr;
  size_t size = 0;
  uint8_t quant[128] = {};
  uint8_t tables = 0;
  uint8_t type = 0;
  uint16_t width = 0, height = 0;
};

inline bool parseJpeg(const uint8_t* data, size_t size, JpegScan& out) {
  out = JpegScan{};
  if (size < 4 || data[0] != 0xff || data[1] != 0xd8) return false;
  size_t pos = 2;
  bool baseline = false;
  while (pos + 3 < size) {
    if (data[pos++] != 0xff) return false;
    while (pos < size && data[pos] == 0xff) ++pos;
    if (pos >= size) return false;
    uint8_t marker = data[pos++];
    if (marker == 0xd9) return false;
    if (pos + 2 > size) return false;
    size_t len = (size_t(data[pos]) << 8) | data[pos+1];
    if (len < 2 || len > size-pos) return false;
    const uint8_t* body = data+pos+2;
    size_t n = len-2;
    if (marker == 0xdb) {
      size_t at = 0;
      while (at < n) {
        uint8_t info = body[at++];
        if ((info >> 4) != 0 || (info & 15) > 1 || n-at < 64) return false;
        uint8_t id = info & 15;
        memcpy(out.quant+id*64, body+at, 64);
        out.tables |= 1 << id;
        at += 64;
      }
    } else if (marker == 0xc0) {
      if (n != 15 || body[0] != 8 || body[5] != 3) return false;
      out.height = (body[1] << 8) | body[2];
      out.width = (body[3] << 8) | body[4];
      // RTP/JPEG supports YUV 4:2:2 (type 0) and 4:2:0 (type 1).
      if ((body[7] != 0x21 && body[7] != 0x22) || body[10] != 0x11 || body[13] != 0x11) return false;
      if (body[8] != 0 || body[11] != 1 || body[14] != 1) return false;
      out.type = body[7] == 0x22 ? 1 : 0;
      baseline = true;
    } else if (marker == 0xc2 || marker == 0xdd) {
      // Progressive JPEG / restart markers require different RTP handling.
      return false;
    } else if (marker == 0xda) {
      if (!baseline || out.tables != 3 || !out.width || !out.height || out.width > 2040 || out.height > 2040) return false;
      pos += len;
      size_t end = size;
      // Camera buffers may include EOI padding. Require a real EOI marker.
      while (end > pos+1 && !(data[end-2] == 0xff && data[end-1] == 0xd9)) --end;
      if (end <= pos+1) return false;
      out.scan = data+pos;
      out.size = end-pos-2;
      return out.size > 0;
    }
    pos += len;
  }
  return false;
}

// One interleaved RTP packet. Returns byte count and advances offset/sequence.
inline size_t jpegPacket(const JpegScan& jpeg, size_t& offset, uint16_t& sequence,
                         uint32_t timestamp, uint32_t ssrc, uint8_t channel,
                         uint8_t* packet, size_t capacity) {
  if (offset >= jpeg.size) return 0;
  size_t extra = offset == 0 ? 132 : 0;
  if (capacity <= 24+extra) return 0;
  size_t chunk = capacity-24-extra;
  if (chunk > jpeg.size-offset) chunk = jpeg.size-offset;
  size_t length = 20+extra+chunk;
  packet[0] = '$'; packet[1] = channel;
  packet[2] = length >> 8; packet[3] = length;
  uint8_t* r = packet+4;
  r[0] = 0x80; r[1] = 26 | (offset+chunk == jpeg.size ? 0x80 : 0);
  r[2] = sequence >> 8; r[3] = sequence++;
  r[4] = timestamp >> 24; r[5] = timestamp >> 16; r[6] = timestamp >> 8; r[7] = timestamp;
  r[8] = ssrc >> 24; r[9] = ssrc >> 16; r[10] = ssrc >> 8; r[11] = ssrc;
  r[12] = 0; r[13] = offset >> 16; r[14] = offset >> 8; r[15] = offset;
  r[16] = jpeg.type; r[17] = 255;
  r[18] = (jpeg.width+7)/8; r[19] = (jpeg.height+7)/8;
  if (!offset) {
    r[20] = 0; r[21] = 0; r[22] = 0; r[23] = 128;
    memcpy(r+24, jpeg.quant, 128);
  }
  memcpy(r+20+extra, jpeg.scan+offset, chunk);
  offset += chunk;
  return length+4;
}
