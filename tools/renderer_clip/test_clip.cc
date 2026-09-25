#include "DiagnosticHomogeneousClip.hh"
#include <iostream>
#include <limits>
using namespace diagnostic_clip;
void require(bool ok) { if (!ok) throw std::runtime_error("Clip assertion failed"); }
int main() {
  const Polygon inside{{-.5,-.5,0,1},{.5,-.5,0,1},{0,.5,0,1}};
  require(clip(inside)==inside);
  require(clip({{2,0,0,1},{3,0,0,1},{2,1,0,1}}).empty());
  require(clip({{0,0,0,-1},{.1,0,0,-1},{0,.1,0,-1}}).empty());
  require(clip({{0,0,-2,1},{.1,0,-2,1},{0,.1,-2,1}}).empty());
  auto crossing=clip({{-2,-.5,0,1},{2,-.5,0,1},{0,2,0,1}});
  require(!crossing.empty());
  auto near=clip({{0,0,-2,1},{.5,0,0,1},{0,.5,0,1}});
  require(near.size()==4);
  auto camera=clip({{0,0,-1,-.5},{.5,0,.1,1},{0,.5,.1,1}});
  require(!camera.empty());
  for (const auto &polygon : {crossing,near,camera}) for (const auto &p:polygon) {
    require(p[3]>0);
    for (unsigned plane=0;plane<6;++plane) require(distance(p,plane)>=-1e-10);
  }
  require(clip({}).empty());
  bool rejected=false;
  try {clip({{std::numeric_limits<double>::quiet_NaN(),0,0,1}});} catch (...) {rejected=true;}
  require(rejected);
  std::cout<<"9 clipping checks passed\n";
}
