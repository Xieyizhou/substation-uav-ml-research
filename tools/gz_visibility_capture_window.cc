#include <gz/transport/Node.hh>
#include <gz/msgs/image.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/annotated_axis_aligned_2d_box_v.pb.h>
#include <google/protobuf/util/json_util.h>
#include <chrono>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>

template<class T> double stamp(const T &m) {
  return m.header().stamp().sec()+m.header().stamp().nsec()*1e-9;
}
template<class T> void write_json(const T &m,const std::string &path) {
  std::string s;auto status=google::protobuf::util::MessageToJsonString(m,&s);
  if(!status.ok()) throw std::runtime_error("JSON encoding failed");
  std::ofstream(path)<<s;
}
int main(int argc,char **argv) {
  if(argc!=3)return 2;
  std::string out=argv[1],world=argv[2];std::filesystem::create_directories(out);
  gz::transport::Node node;std::mutex mutex;
  std::deque<gz::msgs::Image> rgb,depth,mask;
  std::deque<gz::msgs::Pose_V> poses;
  std::deque<gz::msgs::AnnotatedAxisAligned2DBox_V> boxes;
  auto receive=[&](auto &q,const auto &m){std::lock_guard<std::mutex> lock(mutex);q.push_back(m);while(q.size()>8)q.pop_front();};
  std::function<void(const gz::msgs::Image&)> rc=[&](const auto&m){receive(rgb,m);},dc=[&](const auto&m){receive(depth,m);},mc=[&](const auto&m){receive(mask,m);};
  std::function<void(const gz::msgs::Pose_V&)> pc=[&](const auto&m){receive(poses,m);};
  std::function<void(const gz::msgs::AnnotatedAxisAligned2DBox_V&)> bc=[&](const auto&m){receive(boxes,m);};
  if(!node.Subscribe("/research_camera/image",rc)||!node.Subscribe("/research_camera/depth",dc)||!node.Subscribe("/diagnostic/instances/labels_map",mc)||!node.Subscribe("/research_camera/boxes",bc)||!node.Subscribe("/world/"+world+"/pose/info",pc))return 3;
  auto start=std::chrono::steady_clock::now();int count=0;double last=-1;
  while(std::chrono::steady_clock::now()-start<std::chrono::seconds(60)&&count<12){
    std::this_thread::sleep_for(std::chrono::milliseconds(10));std::lock_guard<std::mutex> lock(mutex);
    for(const auto&r:rgb){
      if(stamp(r)<=last)continue;
      auto nearest=[&](const auto&q){auto best=q.end();double delta=.033334001;for(auto i=q.begin();i!=q.end();++i){double d=std::abs(stamp(*i)-stamp(r));if(d<delta){delta=d;best=i;}}return best;};
      auto d=nearest(depth);auto m=nearest(mask);auto p=nearest(poses);auto b=nearest(boxes);
      if(d==depth.end()||m==mask.end()||p==poses.end()||b==boxes.end())continue;
      std::string prefix=out+"/frame-"+std::to_string(++count);
      for(auto pair:{std::make_pair(&r,"rgb"),std::make_pair(&*d,"depth"),std::make_pair(&*m,"mask")}){
        auto metadata=*pair.first;auto data=metadata.data();metadata.clear_data();
        write_json(metadata,prefix+"-"+pair.second+".json");
        std::ofstream f(prefix+"-"+pair.second+".bin",std::ios::binary);f.write(data.data(),data.size());
      }
      write_json(*p,prefix+"-pose.json");write_json(*b,prefix+"-boxes.json");
      last=stamp(r);std::cout<<count<<std::endl;if(count==12)break;
    }
  }
  return count==12?0:4;
}
