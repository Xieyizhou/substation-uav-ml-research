#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include <gz/msgs/image.pb.h>
#include <gz/transport/Node.hh>

struct Sample {
  double timestamp;
};

struct State {
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<Sample> rgb;
  std::deque<Sample> depth;
  std::vector<double> skews_ms;
  std::size_t rgb_count = 0;
  std::size_t depth_count = 0;
  std::size_t pair_count = 0;
  std::size_t dropped_rgb = 0;
  std::size_t dropped_depth = 0;
  int rgb_width = 0;
  int rgb_height = 0;
  int depth_width = 0;
  int depth_height = 0;
  int depth_pixel_format = -1;
  double depth_valid_ratio = -1.0;
};

double Timestamp(const gz::msgs::Image &message) {
  const auto &stamp = message.header().stamp();
  return static_cast<double>(stamp.sec()) +
         static_cast<double>(stamp.nsec()) / 1e9;
}

void Pair(State &state, double maximum_skew_s) {
  while (!state.rgb.empty() && !state.depth.empty()) {
    const double difference =
        state.rgb.front().timestamp - state.depth.front().timestamp;
    if (std::abs(difference) <= maximum_skew_s) {
      state.skews_ms.push_back(std::abs(difference) * 1000.0);
      state.rgb.pop_front();
      state.depth.pop_front();
      ++state.pair_count;
      state.changed.notify_all();
    } else if (difference < 0) {
      state.rgb.pop_front();
      ++state.dropped_rgb;
    } else {
      state.depth.pop_front();
      ++state.dropped_depth;
    }
  }
}

double Percentile(std::vector<double> values, double quantile) {
  if (values.empty()) return -1.0;
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(
      std::max(0.0, std::ceil(quantile * values.size()) - 1.0));
  return values[std::min(index, values.size() - 1)];
}

int main(int argc, char **argv) {
  const int target = argc > 1 ? std::stoi(argv[1]) : 100;
  const double timeout_s = argc > 2 ? std::stod(argv[2]) : 20.0;
  const double maximum_skew_ms = argc > 3 ? std::stod(argv[3]) : 33.334;
  State state;
  gz::transport::Node node;
  std::function<void(const gz::msgs::Image &)> rgb_callback =
      [&](const gz::msgs::Image &message) {
        std::lock_guard<std::mutex> lock(state.mutex);
        ++state.rgb_count;
        state.rgb_width = message.width();
        state.rgb_height = message.height();
        state.rgb.push_back({Timestamp(message)});
        Pair(state, maximum_skew_ms / 1000.0);
      };
  std::function<void(const gz::msgs::Image &)> depth_callback =
      [&](const gz::msgs::Image &message) {
        std::lock_guard<std::mutex> lock(state.mutex);
        ++state.depth_count;
        state.depth_width = message.width();
        state.depth_height = message.height();
        state.depth_pixel_format = message.pixel_format_type();
        if (state.depth_valid_ratio < 0 && message.data().size() % sizeof(float) == 0) {
          const float *values =
              reinterpret_cast<const float *>(message.data().data());
          const std::size_t count = message.data().size() / sizeof(float);
          std::size_t valid = 0;
          for (std::size_t index = 0; index < count; ++index) {
            if (std::isfinite(values[index]) && values[index] >= 0.2f &&
                values[index] <= 100.0f) {
              ++valid;
            }
          }
          state.depth_valid_ratio =
              count ? static_cast<double>(valid) / count : 0.0;
        }
        state.depth.push_back({Timestamp(message)});
        Pair(state, maximum_skew_ms / 1000.0);
      };
  const bool rgb_subscribed =
      node.Subscribe<gz::msgs::Image>("/research_camera/image", rgb_callback);
  const bool depth_subscribed =
      node.Subscribe<gz::msgs::Image>("/research_camera/depth", depth_callback);
  if (!rgb_subscribed || !depth_subscribed) {
    std::cerr << "subscription failed" << std::endl;
    return 2;
  }
  {
    std::unique_lock<std::mutex> lock(state.mutex);
    state.changed.wait_for(
        lock, std::chrono::duration<double>(timeout_s),
        [&] { return state.pair_count >= static_cast<std::size_t>(target); });
  }
  std::lock_guard<std::mutex> lock(state.mutex);
  const double rgb_rate =
      state.rgb_count ? static_cast<double>(state.pair_count) / state.rgb_count : 0;
  const double depth_rate =
      state.depth_count ? static_cast<double>(state.pair_count) / state.depth_count : 0;
  std::cout << std::fixed << std::setprecision(6)
            << "{\"rgb_frame_count\":" << state.rgb_count
            << ",\"depth_frame_count\":" << state.depth_count
            << ",\"pair_count\":" << state.pair_count
            << ",\"dropped_rgb_count\":" << state.dropped_rgb
            << ",\"dropped_depth_count\":" << state.dropped_depth
            << ",\"rgb_pairing_success_rate\":" << rgb_rate
            << ",\"depth_pairing_success_rate\":" << depth_rate
            << ",\"skew_ms\":{\"p50\":"
            << Percentile(state.skews_ms, 0.50)
            << ",\"p95\":" << Percentile(state.skews_ms, 0.95)
            << ",\"max\":" << Percentile(state.skews_ms, 1.0)
            << "},\"rgb_size\":[" << state.rgb_width << "," << state.rgb_height
            << "],\"depth_size\":[" << state.depth_width << ","
            << state.depth_height << "],\"depth_pixel_format\":"
            << state.depth_pixel_format << ",\"depth_valid_ratio\":"
            << state.depth_valid_ratio << "}" << std::endl;
  return state.pair_count >= static_cast<std::size_t>(target) ? 0 : 1;
}
