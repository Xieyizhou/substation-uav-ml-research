#include <gz/sim/Server.hh>
#include <gz/sim/ServerConfig.hh>
int main(int argc, char **argv) {
  if (argc != 2) return 2;
  gz::sim::ServerConfig config;
  if (!config.SetSdfFile(argv[1])) return 3;
  gz::sim::Server server(config);
  return server.Run(true, 0, false) ? 0 : 4;
}
