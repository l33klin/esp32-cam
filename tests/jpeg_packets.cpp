#include "jpeg_rtp.h"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
  assert(argc == 2);
  std::ifstream input(argv[1], std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(input)), {});
  JpegScan jpeg;
  assert(parseJpeg(data.data(), data.size(), jpeg));
  assert(jpeg.width > 0 && jpeg.height > 0);
  JpegScan invalid;
  assert(!parseJpeg(data.data(), data.size()/2, invalid));
  size_t offset = 0;
  uint16_t sequence = 65534;
  uint8_t packet[1404];
  while (offset < jpeg.size) {
    size_t n = jpegPacket(jpeg, offset, sequence, 90000, 0x12345678, 0, packet, sizeof(packet));
    assert(n > 24 && n <= sizeof(packet));
    fwrite(packet, 1, n, stdout);
  }
  return 0;
}
