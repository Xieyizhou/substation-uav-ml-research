#include <gz/sim/Server.hh>
#include <gz/sim/ServerConfig.hh>
#include <mach-o/dyld.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <set>
#include <thread>
int main(int argc,char **argv) {
  if(argc!=2)return 2;
  gz::sim::ServerConfig config;
  if(!config.SetSdfFile(argv[1]))return 3;
  std::atomic<bool> done{false};
  std::thread inventory([&]{
    std::set<std::string> seen;
    while(!done) {
      for(uint32_t i=0;i<_dyld_image_count();++i) {
        std::string p=_dyld_get_image_name(i);
        if(p.find("libgz-rendering8")!=std::string::npos && seen.insert(p).second)
          std::cout<<"DIAGNOSTIC_LOADED_LIBRARY "<<p<<std::endl;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  });
  gz::sim::Server server(config);
  bool ok=server.Run(true,0,false);
  done=true;inventory.join();return ok?0:4;
}
