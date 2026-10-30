#include "rtx_native_api.h"

#include "../dxvk_barrier.h"
#include "rtx_scene_manager.h"

namespace dxvk {

  void NativeRemixApi::initialize(
      const Rc<DxvkDevice>& device,
      const Rc<RtxContext>& context) {
    m_device = device;
    m_context = context;
  }

  void NativeRemixApi::shutdown() {
    m_context = nullptr;
    m_device = nullptr;
  }

  void NativeRemixApi::processExternalCamera(
      CameraType::Enum type,
      const Matrix4& worldToView,
      const Matrix4& viewToProjection) {
    if (!isInitialized())
      return;

    m_device->getCommon()->getSceneManager().getCameraManager()
      .processExternalCamera(type, worldToView, viewToProjection);
  }

  bool NativeRemixApi::registerExternalMesh(
      remixapi_MeshHandle handle,
      std::vector<RasterGeometry> geometries) {
    if (!isInitialized() || !handle || geometries.empty())
      return false;

    auto* assets = m_device->getCommon()->getSceneManager().getAssetReplacer().get();
    assets->registerExternalMesh(handle, std::move(geometries));
    const auto registered = assets->accessExternalMesh(handle);
    return registered != nullptr && !registered->empty();
  }

  bool NativeRemixApi::registerExternalMaterial(
      remixapi_MaterialHandle handle,
      MaterialData material) {
    if (!isInitialized() || !handle)
      return false;

    auto* assets = m_device->getCommon()->getSceneManager().getAssetReplacer().get();
    assets->makeMaterialWithTexturePreload(*m_context, handle, std::move(material));
    return assets->accessExternalMaterial(handle) != nullptr;
  }

  void NativeRemixApi::destroyExternalMesh(remixapi_MeshHandle handle) {
    if (!isInitialized() || !handle)
      return;

    auto& sceneManager = m_device->getCommon()->getSceneManager();
    sceneManager.destroyExternalMesh(
      Rc<DxvkContext>(m_context.ptr()), handle);
  }

  uint32_t NativeRemixApi::commitExternalGeometry(
      std::unique_ptr<ExternalDrawState> state) {
    if (!isInitialized() || !state)
      return 0;

    auto& instances = m_device->getCommon()->getSceneManager().getInstanceManager();
    const uint32_t before = instances.getActiveCount();
    m_context->commitExternalGeometryToRT(std::move(state));
    return instances.getActiveCount() - before;
  }

  void NativeRemixApi::prepareScene() {
    if (!isInitialized())
      return;

    DxvkBarrierSet sceneBarriers(DxvkCmdBuffer::ExecBuffer);
    auto& sceneManager = m_device->getCommon()->getSceneManager();
    sceneManager.prepareSceneData(m_context, sceneBarriers);
    sceneBarriers.recordCommands(m_context->getCommandList());
    m_context->flushCommandList();
  }

}
