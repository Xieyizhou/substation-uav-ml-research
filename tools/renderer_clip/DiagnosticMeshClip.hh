#pragma once
#include "DiagnosticHomogeneousClip.hh"
#include <Vao/OgreIndexBufferPacked.h>
#include <Vao/OgreAsyncTicket.h>
namespace diagnostic_clip {
inline void meshBounds(const Ogre::MeshPtr mesh,
    const Ogre::Matrix4 &view, const Ogre::Matrix4 &projection,
    Ogre::Vector3 &low, Ogre::Vector3 &high,
    const Ogre::Vector3 &position, const Ogre::Quaternion &rotation,
    const Ogre::Vector3 &scale) {
  low=Ogre::Vector3(std::numeric_limits<float>::max());
  high=Ogre::Vector3(-std::numeric_limits<float>::max());
  for (auto sub : mesh->getSubMeshes()) {
    if (sub->mVao[0].empty()) continue;
    auto vao=sub->mVao[0][0];
    if (vao->getOperationType()!=Ogre::OT_TRIANGLE_LIST)
      throw std::runtime_error("Diagnostic clip only supports triangle lists");
    Ogre::VertexArrayObject::ReadRequestsArray requests;
    requests.push_back(Ogre::VertexArrayObject::ReadRequests(Ogre::VES_POSITION));
    vao->readRequests(requests); vao->mapAsyncTickets(requests);
    struct VertexGuard {
      Ogre::VertexArrayObject::ReadRequestsArray &requests;
      ~VertexGuard(){Ogre::VertexArrayObject::unmapAsyncTickets(requests);}
    };
    Polygon vertices;
    {
      VertexGuard guard{requests};
      const auto count=requests[0].vertexBuffer->getNumElements();
      vertices.reserve(count);
      for (size_t i=0;i<count;++i) {
        Ogre::Vector3 v;
        if (requests[0].type==Ogre::VET_HALF4) {
          auto p=reinterpret_cast<const Ogre::uint16*>(requests[0].data);
          v=Ogre::Vector3(Ogre::Bitwise::halfToFloat(p[0]),Ogre::Bitwise::halfToFloat(p[1]),Ogre::Bitwise::halfToFloat(p[2]));
        } else if (requests[0].type==Ogre::VET_FLOAT3) {
          auto p=reinterpret_cast<const float*>(requests[0].data);
          v=Ogre::Vector3(p[0],p[1],p[2]);
        } else throw std::runtime_error("Unsupported diagnostic vertex format");
        v=rotation*(v*scale)+position;
        const auto p=projection*view*Ogre::Vector4(v.x,v.y,v.z,1);
        vertices.push_back({p.x,p.y,p.z,p.w});
        requests[0].data+=requests[0].vertexBuffer->getBytesPerElement();
      }
    }
    const size_t start=vao->getPrimitiveStart(), count=vao->getPrimitiveCount();
    if (count%3) throw std::runtime_error("Incomplete triangle list");
    std::vector<size_t> indices;
    auto index=vao->getIndexBuffer();
    if (index) {
      if (start+count>index->getNumElements()) throw std::runtime_error("Index range overflow");
      auto ticket=index->readRequest(start,count);
      const auto data=ticket->map();
      struct IndexGuard {Ogre::AsyncTicketPtr ticket; ~IndexGuard(){ticket->unmap();}} guard{ticket};
      for (size_t i=0;i<count;++i)
        indices.push_back(index->getIndexType()==Ogre::IT_16BIT ? static_cast<const Ogre::uint16*>(data)[i] : static_cast<const Ogre::uint32*>(data)[i]);
    } else {
      if (start+count>vertices.size()) throw std::runtime_error("Vertex range overflow");
      for (size_t i=start;i<start+count;++i) indices.push_back(i);
    }
    for (size_t i=0;i<indices.size();i+=3) {
      const auto polygon=clip({vertices.at(indices[i]),vertices.at(indices[i+1]),vertices.at(indices[i+2])});
      for (const auto &p:polygon) {
        const Ogre::Vector3 projected(p[0]/p[3],p[1]/p[3],p[2]);
        low.makeFloor(projected); high.makeCeil(projected);
      }
    }
  }
}
}
