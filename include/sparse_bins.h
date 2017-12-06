#ifndef RECOVA_SPARSE_BINS_H
#define RECOVA_SPARSE_BINS_H

#include <array>
#include <map>

namespace recova {
  template<typename T, int N>
  class SparseBins {
  public:
    SparseBins(const T& default_value);
    T get(const std::array<int, N>& coordinates);
    void set(const std::array<int, N>& coordinates, const T& value);
  private:
    T default_value;
    std::map<std::array<int, N>, T> bins;
  };
}

#endif
