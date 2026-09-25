#include "DiagnosticHomogeneousClip.hh"
#include <iostream>
#include <limits>
using namespace diagnostic_clip;
void require(bool ok){if(!ok)throw std::runtime_error("Depth clip assertion failed");}
int main(){
  Polygon inside{{-.5,-.5,0,1},{.5,-.5,0,1},{0,.5,0,1}};
  require(clip(inside)==inside);
  Polygon lateral{{-3,0,0,1},{2,0,0,1},{0,3,0,1}};
  require(clip(lateral)==lateral);
  require(clip({{0,0,-2,1},{.1,0,-2,1},{0,.1,-2,1}}).empty());
  require(clip({{0,0,2,1},{.1,0,2,1},{0,.1,2,1}}).empty());
  require(clip({{0,0,0,-1},{.1,0,0,-1},{0,.1,0,-1}}).empty());
  auto near=clip({{0,0,-2,1},{2,0,0,1},{0,2,0,1}});
  require(near.size()==4);
  for(const auto &p:near){require(p[3]>0);require(distance(p,4)>=-1e-12);require(distance(p,5)>=-1e-12);}
  require(clip({}).empty());
  bool rejected=false;try{clip({{std::numeric_limits<double>::infinity(),0,0,1}});}catch(...){rejected=true;}
  require(rejected);
  std::cout<<"8 depth-plane clipping checks passed\n";
}
