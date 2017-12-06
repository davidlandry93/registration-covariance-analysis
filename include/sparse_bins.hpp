#ifndef RECOVA_SPARSE_BINS_HPP
#define RECOVA_SPARSE_BINS_HPP

#include "sparse_bins.h"

namespace recova {
  template<typename T, int N>
  SparseBins<T, N>::SparseBins(const T& default_value) : default_value(default_value) {}

  template<typename T, int N>
  T SparseBins<T,N>::get(const std::array<int, N>& coordinates) {
    auto it = bins.find(coordinates);

    if(it == bins.end()) {
      return default_value;
    } else {
      return it->second;
    }
  }

  template<typename T, int N>
  void SparseBins<T,N>::set(const std::array<int, N>& coordinates, const T& value) {
    bins[coordinates] = value;
  }
}

#endif
