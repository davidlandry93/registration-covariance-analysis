#ifndef RECOVA_SPARSE_BINS_H
#define RECOVA_SPARSE_BINS_H

#include <array>
#include <map>
#include <vector>

namespace recova {
  template<typename T, int N>
  class SparseBins {
  public:
    std::vector<T> get(const std::array<int, N>& coordinates);
    void add_to_bin(const std::array<int, N>& coordinates, const T& value);

  private:
    std::map<std::array<int, N>, std::vector<T>> bins;
  };
}

#endif
