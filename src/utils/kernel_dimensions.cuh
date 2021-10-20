int nextPow2(int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

void getNumBlocksAndThreads(const int n, const int maxBlockSize, int &blocks,
                            int &threads) {
  threads = (n < maxBlockSize * 2) ? nextPow2((n + 1) / 2) : maxBlockSize;
  blocks = (n + threads - 1) / threads;
}
