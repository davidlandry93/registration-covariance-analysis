#ifndef RECOVA_SPARSE_BINS_HPP
#define RECOVA_SPARSE_BINS_HPP

#include "sparse_bins.h"

namespace recova {
  template<typename T, int N>
  std::vector<T> SparseBins<T,N>::get(const std::array<int, N>& coordinates) {
    return bins[coordinates];
  }

  template<typename T, int N>
  void SparseBins<T,N>::add_to_bin(const std::array<int, N>& coordinates, const T& value) {
    bins[coordinates].push_back(value);
  }
}

#endif
