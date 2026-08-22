#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include <gz/msgs/image.pb.h>
#include <gz/transport/Node.hh>

struct Frame {
  gz::msgs::Image message;
  double timestamp;
};

struct Bridge {
  std::mutex mutex;
  std::condition_variable stopped;
  std::deque<Frame> rgb;
  std::deque<Frame> depth;
  std::vector<double> skews_ms;
  std::filesystem::path output;
  std::size_t rgb_count = 0;
  std::size_t depth_count = 0;
  std::size_t pair_count = 0;
  std::size_t dropped_rgb = 0;
  std::size_t dropped_depth = 0;
  int emit_stride = 6;
  double maximum_skew_s = 0.033334;
};

double Timestamp(const gz::msgs::Image &message) {
  const auto &stamp = message.header().stamp();
  return static_cast<double>(stamp.sec()) +
         static_cast<double>(stamp.nsec()) / 1e9;
}

double Percentile(std::vector<double> values, double quantile) {
  if (values.empty()) return -1.0;
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(
      std::max(0.0, std::ceil(quantile * values.size()) - 1.0));
  return values[std::min(index, values.size() - 1)];
}

void WritePair(Bridge &bridge, const Frame &rgb, const Frame &depth,
               double skew_ms) {
  const std::string stem = "pair-" + std::to_string(bridge.pair_count);
  const auto rgb_path = bridge.output / (stem + ".ppm");
  const auto depth_path = bridge.output / (stem + ".depth");
  const auto &rgb_data = rgb.message.data();
  const std::size_t expected_rgb =
      static_cast<std::size_t>(rgb.message.width()) * rgb.message.height() * 3;
  if (rgb_data.size() != expected_rgb) {
    std::cerr << "unexpected RGB payload size: " << rgb_data.size() << std::endl;
    return;
  }
  {
    std::ofstream stream(rgb_path, std::ios::binary);
    stream << "P6\n" << rgb.message.width() << " " << rgb.message.height()
           << "\n255\n";
    stream.write(rgb_data.data(), static_cast<std::streamsize>(rgb_data.size()));
  }
  {
    std::ofstream stream(depth_path, std::ios::binary);
    stream.write(depth.message.data().data(),
                 static_cast<std::streamsize>(depth.message.data().size()));
  }
  const double rgb_rate =
      bridge.rgb_count
          ? static_cast<double>(bridge.pair_count) / bridge.rgb_count
          : 0;
  const double depth_rate =
      bridge.depth_count
          ? static_cast<double>(bridge.pair_count) / bridge.depth_count
          : 0;
  std::cout << std::fixed << std::setprecision(6)
            << "{\"pair_count\":" << bridge.pair_count
            << ",\"rgb_frame_count\":" << bridge.rgb_count
            << ",\"depth_frame_count\":" << bridge.depth_count
            << ",\"dropped_rgb_count\":" << bridge.dropped_rgb
            << ",\"dropped_depth_count\":" << bridge.dropped_depth
            << ",\"rgb_pairing_success_rate\":" << rgb_rate
            << ",\"depth_pairing_success_rate\":" << depth_rate
            << ",\"skew_ms\":" << skew_ms
            << ",\"skew_p50_ms\":"
            << Percentile(bridge.skews_ms, 0.50)
            << ",\"skew_p95_ms\":"
            << Percentile(bridge.skews_ms, 0.95)
            << ",\"skew_max_ms\":"
            << Percentile(bridge.skews_ms, 1.0)
            << ",\"rgb_timestamp\":" << rgb.timestamp
            << ",\"depth_timestamp\":" << depth.timestamp
            << ",\"rgb_width\":" << rgb.message.width()
            << ",\"rgb_height\":" << rgb.message.height()
            << ",\"depth_width\":" << depth.message.width()
            << ",\"depth_height\":" << depth.message.height()
            << ",\"rgb_path\":\"" << rgb_path.filename().string()
            << "\",\"depth_path\":\"" << depth_path.filename().string()
            << "\"}" << std::endl;
}

void Pair(Bridge &bridge) {
  while (!bridge.rgb.empty() && !bridge.depth.empty()) {
    const double difference =
        bridge.rgb.front().timestamp - bridge.depth.front().timestamp;
    if (std::abs(difference) <= bridge.maximum_skew_s) {
      Frame rgb = std::move(bridge.rgb.front());
      Frame depth = std::move(bridge.depth.front());
      bridge.rgb.pop_front();
      bridge.depth.pop_front();
      ++bridge.pair_count;
      const double skew_ms = std::abs(difference) * 1000.0;
      bridge.skews_ms.push_back(skew_ms);
      if (bridge.pair_count % bridge.emit_stride == 0) {
        WritePair(bridge, rgb, depth, skew_ms);
      }
    } else if (difference < 0) {
      bridge.rgb.pop_front();
      ++bridge.dropped_rgb;
    } else {
      bridge.depth.pop_front();
      ++bridge.dropped_depth;
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: gz_rgbd_bridge OUTPUT [EMIT_STRIDE] [MAX_SKEW_MS]"
              << std::endl;
    return 2;
  }
  Bridge bridge;
  bridge.output = argv[1];
  bridge.emit_stride = argc > 2 ? std::max(1, std::stoi(argv[2])) : 6;
  bridge.maximum_skew_s =
      (argc > 3 ? std::stod(argv[3]) : 33.334) / 1000.0;
  std::filesystem::create_directories(bridge.output);
  gz::transport::Node node;
  std::function<void(const gz::msgs::Image &)> rgb_callback =
      [&](const gz::msgs::Image &message) {
        std::lock_guard<std::mutex> lock(bridge.mutex);
        ++bridge.rgb_count;
        bridge.rgb.push_back({message, Timestamp(message)});
        Pair(bridge);
      };
  std::function<void(const gz::msgs::Image &)> depth_callback =
      [&](const gz::msgs::Image &message) {
        std::lock_guard<std::mutex> lock(bridge.mutex);
        ++bridge.depth_count;
        bridge.depth.push_back({message, Timestamp(message)});
        Pair(bridge);
      };
  if (!node.Subscribe<gz::msgs::Image>("/research_camera/image", rgb_callback) ||
      !node.Subscribe<gz::msgs::Image>("/research_camera/depth", depth_callback)) {
    std::cerr << "RGB-D subscription failed" << std::endl;
    return 3;
  }
  std::unique_lock<std::mutex> lock(bridge.mutex);
  bridge.stopped.wait(lock);
  return 0;
}
