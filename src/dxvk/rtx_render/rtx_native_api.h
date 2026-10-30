#pragma once

#include "../dxvk_device.h"
#include "rtx_context.h"
#include "rtx_types.h"

#include <memory>
#include <vector>

namespace dxvk {

  // Native Remix access used by non-D3D9 frontends. This facade owns no
  // device; its references are valid only while the caller keeps the device
  // and RtxContext alive.
  class NativeRemixApi {
  public:
    NativeRemixApi() = default;

    NativeRemixApi(const NativeRemixApi&) = delete;
    NativeRemixApi& operator=(const NativeRemixApi&) = delete;

    void initialize(const Rc<DxvkDevice>& device, const Rc<RtxContext>& context);
    void shutdown();

    bool isInitialized() const { return m_device != nullptr && m_context != nullptr; }

    DxvkDevice* getDevice() const { return m_device.ptr(); }
    RtxContext* getContext() const { return m_context.ptr(); }

    void processExternalCamera(
        CameraType::Enum type,
        const Matrix4& worldToView,
        const Matrix4& viewToProjection);

    bool registerExternalMesh(
        remixapi_MeshHandle handle,
        std::vector<RasterGeometry> geometries);

    bool registerExternalMaterial(
        remixapi_MaterialHandle handle,
        MaterialData material);

    void destroyExternalMesh(remixapi_MeshHandle handle);

    uint32_t commitExternalGeometry(std::unique_ptr<ExternalDrawState> state);

    void prepareScene();

  private:
    Rc<DxvkDevice> m_device;
    Rc<RtxContext> m_context;
  };

}
