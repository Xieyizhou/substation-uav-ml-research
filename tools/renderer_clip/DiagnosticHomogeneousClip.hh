// Diagnostic-only homogeneous polygon clipping. No runtime admission policy.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace diagnostic_clip {
using Point = std::array<double,4>;
using Polygon = std::vector<Point>;
inline double distance(const Point &p, unsigned plane) {
  return p[3] + (plane%2 ? -p[plane/2] : p[plane/2]);
}
inline Polygon clip(Polygon input) {
  for (const auto &p : input) for (auto x : p)
    if (!std::isfinite(x)) throw std::runtime_error("Nonfinite clip vertex");
  for (unsigned plane=0; plane<6 && !input.empty(); ++plane) {
    Polygon output;
    auto previous=input.back(); auto before=distance(previous,plane);
    for (const auto &current : input) {
      const auto now=distance(current,plane);
      if ((before>=0)!=(now>=0)) {
        const auto t=before/(before-now);
        Point cut{};
        for (unsigned i=0;i<4;++i) cut[i]=previous[i]+t*(current[i]-previous[i]);
        output.push_back(cut);
      }
      if (now>=0) output.push_back(current);
      previous=current; before=now;
    }
    input=std::move(output);
  }
  input.erase(std::remove_if(input.begin(),input.end(),[](const auto &p){return p[3]<=1e-12;}),input.end());
  return input;
}
}
