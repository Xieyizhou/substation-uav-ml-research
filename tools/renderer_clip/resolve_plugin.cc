#include <gz/common/SystemPaths.hh>
#include <iostream>
#include <cstdlib>
#include <gz/rendering/InstallationDirectories.hh>
int main() {
  std::cout<<"env="<<(std::getenv("GZ_RENDERING_PLUGIN_PATH")?std::getenv("GZ_RENDERING_PLUGIN_PATH"):"missing")<<"\n";
  gz::common::SystemPaths p;
  p.SetPluginPathEnv("GZ_RENDERING_PLUGIN_PATH");
  p.AddPluginPaths("/opt/homebrew/Cellar/gz-rendering8/8.2.3_4/lib");
  p.AddPluginPaths(gz::rendering::getEngineInstallDir());
  for(const auto &s:p.PluginPaths())std::cout<<"path="<<s<<"\n";
  std::cout<<"result="<<p.FindSharedLibrary("gz-rendering-ogre2")<<"\n";
}
