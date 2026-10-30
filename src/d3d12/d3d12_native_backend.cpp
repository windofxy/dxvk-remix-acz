#include "d3d12_native_backend.h"

#include "d3d12_command_list_trace.h"
#include "d3d12_device_proxy.h"
#include "d3d12_resource_registry.h"
#include "d3d12_runtime_adapter.h"

#include "../dxvk/dxvk_adapter.h"
#include "../dxvk/dxvk_barrier.h"
#include "../dxvk/dxvk_buffer.h"
#include "../dxvk/dxvk_context.h"
#include "../dxvk/dxvk_device.h"
#include "../dxvk/dxvk_instance.h"
#include "../dxvk/dxvk_image.h"
#include "../dxvk/rtx_render/rtx_asset_replacer.h"
#include "../dxvk/rtx_render/rtx_context.h"
#include "../dxvk/rtx_render/rtx_hashing.h"
#include "../dxvk/rtx_render/rtx_native_api.h"
#include "../dxvk/rtx_render/rtx_scene_manager.h"
#include "../dxvk/rtx_render/rtx_types.h"
#include "../dxvk/rtx_render/rtx_texture.h"
#include "../util/util_error.h"
#include "../util/rc/util_rc_ptr.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <windows.h>
#include <unordered_set>
#include <vector>

dxvk::Logger dxvk::Logger::s_instance("remix-d3d12-remix.log");

namespace remix {

  namespace {
    std::mutex g_backendMutex;
    dxvk::Rc<dxvk::DxvkDevice> g_device;
    dxvk::Rc<dxvk::RtxContext> g_rtxContext;
    dxvk::NativeRemixApi g_nativeRemixApi;
    bool g_initialized = false;
    bool g_failed = false;
    bool g_sceneInitialized = false;
    uint64_t g_sceneMeshSerial = 0;
    std::unordered_set<uint64_t> g_submittedGeometry;
    std::unordered_set<uint64_t> g_registeredMaterials;
    // Once a draw has produced a native mesh, avoid remapping and hashing the
    // same GPU ranges again on every frame.
    std::unordered_set<uint64_t> g_seenGeometryRequests;
    // Bounded diagnostics for correlating anomalous geometry with the source
    // object transform. This is intentionally process-limited to avoid making
    // the per-draw path expensive during normal runs.
    std::atomic<uint32_t> g_transformDiagnosticCount { 0 };
    std::atomic<uint32_t> g_filteredOriginDrawCount { 0 };
    uint64_t g_lastCameraHash = 0;
    uint32_t g_lastCameraFrame = UINT32_MAX;
    dxvk::Matrix4 g_cameraWorldToView;
    dxvk::Matrix4 g_cameraViewToProjection;
    dxvk::Vector3 g_cameraEye;
    std::atomic<float> g_nativeNearestInstanceDistanceSq { 3.4e38f };
    std::atomic<uint32_t> g_nativeNearInstances { 0 };
    bool g_hasCameraMatrices = false;
    bool g_hasPreparedScene = false;
    uint32_t g_preservedSceneFrames = 0;
    ID3D12Device* g_d3d12Device = nullptr;
    ID3D12CommandQueue* g_d3d12Queue = nullptr;
    LARGE_INTEGER g_perfFrequency {};
    std::once_flag g_perfFrequencyOnce;
    std::atomic<uint64_t> g_nativeSubmitTicks { 0 };
    std::atomic<uint64_t> g_nativeSubmitCalls { 0 };
    std::atomic<uint64_t> g_nativePrepareTicks { 0 };
    std::atomic<uint64_t> g_nativePrepareCalls { 0 };
    std::atomic<uint64_t> g_nativeInjectTicks { 0 };
    std::atomic<uint64_t> g_nativeInjectCalls { 0 };
    std::atomic<uint64_t> g_nativeNoTransform { 0 };
    std::atomic<uint64_t> g_nativeIdentityFiltered { 0 };
    // White model mode: every surface is submitted opaque and self lit, so the
    // captured geometry is visible without the game's textures. F7 toggles it.
    std::atomic<uint32_t> g_forceWhiteMaterials { 1 };

    // Vertex shaders that draw object geometry with a per-object matrix. The
    // white model path leaves these lit rather than self lit, because a self lit
    // aircraft and a self lit terrain are the same white and the aircraft
    // disappears into the ground behind it.
    bool isAczModelShader(uint64_t hash) {
      switch (hash) {
      case 0xb6a7d1d096ce8fe7ull:  // DrawACM_v, aircraft
      case 0xeb39ece513987798ull:  // DrawModelGsh_v
      case 0x8a562777e73aa8d5ull:  // DrawModelIcon_v
      case 0x3c10411c7e513ba4ull:  // DrawSelMap_v
      case 0x9fcd88e1b090fcaaull:  // DrawModelShadow_v
        return true;
      default:
        return false;
      }
    }
    // Fate counters for the aircraft draw, so a draw that never reaches Remix can
    // be attributed to one stage instead of being looked for in the frame. Only
    // DrawACM_v is counted; the stages are the exits of the capture path.
    enum AircraftStage : uint32_t {
      kAircraftDraw,
      kAircraftSubmitted,
      kAircraftGated,
      kAircraftTopology,
      kAircraftUnreadable,
      kAircraftIdentity,
      kAircraftNoTransform,
      kAircraftApiRejected,
      kAircraftStageCount,
    };

    std::atomic<uint64_t> g_aircraftStages[kAircraftStageCount];

    void noteAircraftDraw(uint64_t shaderHash, AircraftStage stage) {
      if (shaderHash != 0xb6a7d1d096ce8fe7ull)
        return;
      const uint64_t value = g_aircraftStages[stage].fetch_add(
        1, std::memory_order_relaxed) + 1;
      if (stage == kAircraftDraw && (value % 512) == 0) {
        log("ACZ aircraft verdict: draws=%llu submitted=%llu gated=%llu topology=%llu unreadable=%llu identity=%llu noTransform=%llu apiRejected=%llu",
          static_cast<unsigned long long>(g_aircraftStages[kAircraftDraw].load(std::memory_order_relaxed)),
          static_cast<unsigned long long>(g_aircraftStages[kAircraftSubmitted].load(std::memory_order_relaxed)),
          static_cast<unsigned long long>(g_aircraftStages[kAircraftGated].load(std::memory_order_relaxed)),
          static_cast<unsigned long long>(g_aircraftStages[kAircraftTopology].load(std::memory_order_relaxed)),
          static_cast<unsigned long long>(g_aircraftStages[kAircraftUnreadable].load(std::memory_order_relaxed)),
          static_cast<unsigned long long>(g_aircraftStages[kAircraftIdentity].load(std::memory_order_relaxed)),
          static_cast<unsigned long long>(g_aircraftStages[kAircraftNoTransform].load(std::memory_order_relaxed)),
          static_cast<unsigned long long>(g_aircraftStages[kAircraftApiRejected].load(std::memory_order_relaxed)));
      }
    }

    std::atomic<uint64_t> g_nativeReadFailures { 0 };
    std::atomic<uint64_t> g_nativeConversionFailures { 0 };
    std::atomic<uint64_t> g_nativeAcceptedInstances { 0 };
    std::atomic<uint64_t> g_nativeWorldSpaceFallbacks { 0 };
    std::atomic<uint64_t> g_nativeWorldSpaceSubmits { 0 };
    std::atomic<uint64_t> g_nativeWorldSpaceInstances { 0 };
    // Enables the Remix output path. Disabling it leaves the game's native D3D12
    // frame, including its own 2D UI, on screen while capture keeps running.
    std::atomic<bool> g_remixOutputEnabled { true };
    // F10 toggles the Remix output so the native frame can be inspected while
    // capture keeps running.
    bool g_remixOutputKeyDown = false;
    bool g_whiteMaterialKeyDown = false;
    std::atomic<uint32_t> g_transformRejectionDiagnostics { 0 };

    uint64_t perfTicks() {
      std::call_once(g_perfFrequencyOnce, [] {
        QueryPerformanceFrequency(&g_perfFrequency);
      });
      LARGE_INTEGER value {};
      QueryPerformanceCounter(&value);
      return static_cast<uint64_t>(value.QuadPart);
    }

    struct PerfScope {
      std::atomic<uint64_t>& ticks;
      std::atomic<uint64_t>& calls;
      uint64_t start = perfTicks();
      ~PerfScope() {
        ticks.fetch_add(perfTicks() - start, std::memory_order_relaxed);
        calls.fetch_add(1, std::memory_order_relaxed);
      }
    };

    struct ExternalFrameSync {
      ID3D12Fence* inputFence = nullptr;
      ID3D12Fence* outputFence = nullptr;
      VkSemaphore inputSemaphore = VK_NULL_HANDLE;
      VkSemaphore outputSemaphore = VK_NULL_HANDLE;
      uint64_t inputValue = 0;
      uint64_t outputValue = 0;
      bool initialized = false;
    };

    ExternalFrameSync g_externalFrameSync;
    void* g_outputSwapChain = nullptr;
    ID3D12Resource* g_outputResource = nullptr;
    UINT g_outputBackBufferIndex = UINT_MAX;
    D3D12_RESOURCE_DESC g_outputDescription {};
    dxvk::Rc<dxvk::DxvkImage> g_outputImage;

    struct OutputImportCacheEntry {
      void* swapChain = nullptr;
      ID3D12Resource* resource = nullptr;
      D3D12_RESOURCE_DESC description {};
      dxvk::Rc<dxvk::DxvkImage> image;
      bool attempted = false;
      bool succeeded = false;
    };

    std::array<OutputImportCacheEntry, 8> g_outputImportCache;

    bool createImportedD3D12FenceSemaphore(
        ID3D12Fence* fence,
        VkSemaphore& semaphore,
        const char* name) {
      if (!g_device.ptr() || !fence || !g_device->vkd()->vkImportSemaphoreWin32HandleKHR)
        return false;

      HANDLE handle = nullptr;
      ID3D12Device* device = nullptr;
      if (FAILED(fence->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void**>(&device)))
       || !device)
        return false;
      HRESULT result = device->CreateSharedHandle(
        fence, nullptr, GENERIC_ALL, nullptr, &handle);
      device->Release();
      if (FAILED(result) || !handle)
        return false;

      VkSemaphoreTypeCreateInfo timelineInfo {};
      timelineInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
      timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
      timelineInfo.initialValue = 0;

      VkSemaphoreCreateInfo createInfo {};
      createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
      createInfo.pNext = &timelineInfo;
      if (g_device->vkd()->vkCreateSemaphore(
          g_device->handle(), &createInfo, nullptr, &semaphore) != VK_SUCCESS) {
        CloseHandle(handle);
        return false;
      }

      VkImportSemaphoreWin32HandleInfoKHR importInfo {};
      importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR;
      importInfo.semaphore = semaphore;
      importInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
      importInfo.handle = handle;
      VkResult importResult = g_device->vkd()->vkImportSemaphoreWin32HandleKHR(
        g_device->handle(), &importInfo);
      if (importResult != VK_SUCCESS) {
        g_device->vkd()->vkDestroySemaphore(g_device->handle(), semaphore, nullptr);
        semaphore = VK_NULL_HANDLE;
        CloseHandle(handle);
        log("D3D12 native Remix external fence import failed: name=%s vkResult=%d",
          name ? name : "unknown", static_cast<int>(importResult));
        return false;
      }

      if (g_device->vkd()->vkSetDebugUtilsObjectNameEXT) {
        VkDebugUtilsObjectNameInfoEXT nameInfo {};
        nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        nameInfo.objectType = VK_OBJECT_TYPE_SEMAPHORE;
        nameInfo.objectHandle = reinterpret_cast<uint64_t>(semaphore);
        nameInfo.pObjectName = name;
        g_device->vkd()->vkSetDebugUtilsObjectNameEXT(
          g_device->handle(), &nameInfo);
      }
      // Ownership of an NT handle is transferred by the Vulkan import call.
      return true;
    }

    bool initializeExternalFrameSync() {
      if (g_externalFrameSync.initialized)
        return true;
      if (!g_device.ptr() || !g_d3d12Device || !g_d3d12Queue)
        return false;

      HRESULT result = g_d3d12Device->CreateFence(
        0, D3D12_FENCE_FLAG_SHARED, __uuidof(ID3D12Fence),
        reinterpret_cast<void**>(&g_externalFrameSync.inputFence));
      if (FAILED(result))
        return false;
      result = g_d3d12Device->CreateFence(
        0, D3D12_FENCE_FLAG_SHARED, __uuidof(ID3D12Fence),
        reinterpret_cast<void**>(&g_externalFrameSync.outputFence));
      if (FAILED(result))
        return false;

      if (!createImportedD3D12FenceSemaphore(
          g_externalFrameSync.inputFence,
          g_externalFrameSync.inputSemaphore,
          "D3D12 input fence")
       || !createImportedD3D12FenceSemaphore(
          g_externalFrameSync.outputFence,
          g_externalFrameSync.outputSemaphore,
          "D3D12 output fence")) {
        log("D3D12 native Remix external frame sync unavailable");
        return false;
      }

      g_externalFrameSync.initialized = true;
      log("D3D12 native Remix external frame sync initialized: inputFence=%p outputFence=%p inputSemaphore=%p outputSemaphore=%p",
        g_externalFrameSync.inputFence,
        g_externalFrameSync.outputFence,
        reinterpret_cast<void*>(g_externalFrameSync.inputSemaphore),
        reinterpret_cast<void*>(g_externalFrameSync.outputSemaphore));
      return true;
    }

    constexpr size_t kMaxNativeGeometryBytes = 16 * 1024 * 1024;

    VkFormat toVkOutputFormat(DXGI_FORMAT format) {
      switch (format) {
      case DXGI_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
      case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
      case DXGI_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
      case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return VK_FORMAT_B8G8R8A8_SRGB;
      case DXGI_FORMAT_R10G10B10A2_UNORM: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
      case DXGI_FORMAT_R16G16B16A16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
      default: return VK_FORMAT_UNDEFINED;
      }
    }

    VkSampleCountFlagBits toVkSampleCount(UINT count) {
      switch (count) {
      case 2: return VK_SAMPLE_COUNT_2_BIT;
      case 4: return VK_SAMPLE_COUNT_4_BIT;
      case 8: return VK_SAMPLE_COUNT_8_BIT;
      case 16: return VK_SAMPLE_COUNT_16_BIT;
      default: return VK_SAMPLE_COUNT_1_BIT;
      }
    }

    bool importBackBuffer(
        void* swapChain,
        UINT backBufferIndex,
        ID3D12Resource* backBuffer,
        const D3D12_RESOURCE_DESC* description) {
      if (!g_device.ptr() || !g_rtxContext.ptr() || !swapChain || !backBuffer || !description)
        return false;

      auto sameDescription = [&](const D3D12_RESOURCE_DESC& cached) {
        return cached.Dimension == description->Dimension
          && cached.Width == description->Width
          && cached.Height == description->Height
          && cached.DepthOrArraySize == description->DepthOrArraySize
          && cached.MipLevels == description->MipLevels
          && cached.Format == description->Format
          && cached.SampleDesc.Count == description->SampleDesc.Count
          && cached.SampleDesc.Quality == description->SampleDesc.Quality;
      };

      OutputImportCacheEntry* cacheEntry = nullptr;
      for (auto& candidate : g_outputImportCache) {
        if (candidate.swapChain == swapChain && candidate.resource == backBuffer) {
          if (sameDescription(candidate.description)) {
            cacheEntry = &candidate;
            break;
          }
          candidate = {};
        }
      }
      if (!cacheEntry) {
        for (auto& candidate : g_outputImportCache) {
          if (!candidate.resource) {
            cacheEntry = &candidate;
            break;
          }
        }
      }
      if (!cacheEntry) {
        cacheEntry = &g_outputImportCache[backBufferIndex % g_outputImportCache.size()];
        *cacheEntry = {};
      }

      g_outputSwapChain = swapChain;
      g_outputResource = backBuffer;
      g_outputBackBufferIndex = backBufferIndex;
      g_outputDescription = *description;
      if (cacheEntry->attempted) {
        g_outputImage = cacheEntry->succeeded ? cacheEntry->image : nullptr;
        return cacheEntry->succeeded;
      }
      cacheEntry->swapChain = swapChain;
      cacheEntry->resource = backBuffer;
      cacheEntry->description = *description;
      cacheEntry->attempted = true;

      const VkFormat format = toVkOutputFormat(description->Format);
      if (format == VK_FORMAT_UNDEFINED) {
        log("D3D12 native Remix output import skipped: unsupported DXGI format=%u",
          static_cast<unsigned>(description->Format));
        g_outputImage = nullptr;
        return false;
      }

      ID3D12Device* d3d12Device = nullptr;
      HRESULT deviceResult = backBuffer->GetDevice(
        __uuidof(ID3D12Device), reinterpret_cast<void**>(&d3d12Device));
      if (FAILED(deviceResult) || !d3d12Device) {
        log("D3D12 native Remix output import skipped: GetDevice result=0x%08lx",
          static_cast<unsigned long>(deviceResult));
        return false;
      }

      HANDLE sharedHandle = nullptr;
      HRESULT handleResult = d3d12Device->CreateSharedHandle(
        backBuffer, nullptr, GENERIC_ALL, nullptr, &sharedHandle);
      d3d12Device->Release();
      if (FAILED(handleResult) || !sharedHandle) {
        log("D3D12 native Remix output import unavailable: CreateSharedHandle result=0x%08lx resource=%p",
          static_cast<unsigned long>(handleResult), backBuffer);
        g_outputImage = nullptr;
        return false;
      }

      dxvk::DxvkImageCreateInfo imageInfo {};
      imageInfo.type = VK_IMAGE_TYPE_2D;
      imageInfo.format = format;
      imageInfo.flags = 0;
      imageInfo.sampleCount = toVkSampleCount(description->SampleDesc.Count);
      imageInfo.extent = { static_cast<uint32_t>(description->Width), description->Height, 1 };
      imageInfo.numLayers = 1;
      imageInfo.mipLevels = 1;
      imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT
        | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        | VK_IMAGE_USAGE_SAMPLED_BIT;
      imageInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT
        | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
        | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
      imageInfo.access = VK_ACCESS_TRANSFER_READ_BIT
        | VK_ACCESS_TRANSFER_WRITE_BIT
        | VK_ACCESS_SHADER_READ_BIT;
      imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
      imageInfo.layout = VK_IMAGE_LAYOUT_GENERAL;
      imageInfo.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
      imageInfo.shared = VK_TRUE;
      imageInfo.sharing.mode = dxvk::DxvkSharedHandleMode::Import;
      // ID3D12Device::CreateSharedHandle returns an NT handle. KMT handles
      // are used by the legacy D3D9 path, but importing this D3D12 handle as
      // KMT can appear to work for one buffer and fail allocation for the
      // next swap-chain image.
      imageInfo.sharing.type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
      imageInfo.sharing.handle = sharedHandle;

      try {
        g_outputImage = g_device->createImage(
          imageInfo,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          dxvk::DxvkMemoryStats::Category::RTXRenderTarget,
          "D3D12 imported back buffer");
      } catch (const dxvk::DxvkError& error) {
        log("D3D12 native Remix output import failed: %s", error.message().c_str());
        CloseHandle(sharedHandle);
        g_outputImage = nullptr;
        return false;
      } catch (...) {
        log("D3D12 native Remix output import failed: unknown exception");
        CloseHandle(sharedHandle);
        g_outputImage = nullptr;
        return false;
      }

      if (!g_outputImage.ptr()) {
        log("D3D12 native Remix output import failed: Vulkan image is null");
        CloseHandle(sharedHandle);
        return false;
      }

      cacheEntry->image = g_outputImage;
      cacheEntry->succeeded = true;
      log("D3D12 native Remix output import succeeded: swapChain=%p index=%u resource=%p image=%p",
        swapChain, backBufferIndex, backBuffer, g_outputImage.ptr());
      return true;
    }

    struct NativeMeshStorage {
      remixapi_MeshHandle handle = nullptr;
      dxvk::Rc<dxvk::DxvkBuffer> vertexBuffer;
      dxvk::Rc<dxvk::DxvkBuffer> indexBuffer;
    };

    std::vector<NativeMeshStorage> g_meshStorage;
    std::unordered_map<uint64_t, remixapi_MeshHandle> g_nativeMeshHandles;
    std::unordered_map<uint64_t, remixapi_MeshHandle> g_nativeRequestHandles;
    std::unordered_set<uint64_t> g_nativeWorldSpaceMeshes;
    std::unordered_set<uint64_t> g_nativeWorldSpaceRequests;
    std::unordered_map<uint64_t, dxvk::TextureRef> g_nativeTextureRefs;
    std::unordered_map<uint64_t, dxvk::Rc<dxvk::DxvkSampler>> g_nativeSamplers;
    uint64_t g_nativeTextureBytes = 0;
    constexpr size_t kMaxNativeTextureRefs = 64;
    constexpr uint64_t kMaxNativeTextureBytes = 8ull * 1024ull * 1024ull;
    constexpr bool kEnableNativeTextureMaterials = false;

    VkFormat toVkMaterialFormat(DXGI_FORMAT format) {
      switch (format) {
      case DXGI_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
      case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
      case DXGI_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
      case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return VK_FORMAT_B8G8R8A8_SRGB;
      case DXGI_FORMAT_BC1_UNORM: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
      case DXGI_FORMAT_BC1_UNORM_SRGB: return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
      case DXGI_FORMAT_BC2_UNORM: return VK_FORMAT_BC2_UNORM_BLOCK;
      case DXGI_FORMAT_BC2_UNORM_SRGB: return VK_FORMAT_BC2_SRGB_BLOCK;
      case DXGI_FORMAT_BC3_UNORM: return VK_FORMAT_BC3_UNORM_BLOCK;
      case DXGI_FORMAT_BC3_UNORM_SRGB: return VK_FORMAT_BC3_SRGB_BLOCK;
      case DXGI_FORMAT_BC4_UNORM: return VK_FORMAT_BC4_UNORM_BLOCK;
      case DXGI_FORMAT_BC5_UNORM: return VK_FORMAT_BC5_UNORM_BLOCK;
      case DXGI_FORMAT_BC6H_UF16: return VK_FORMAT_BC6H_UFLOAT_BLOCK;
      case DXGI_FORMAT_BC6H_SF16: return VK_FORMAT_BC6H_SFLOAT_BLOCK;
      case DXGI_FORMAT_BC7_UNORM: return VK_FORMAT_BC7_UNORM_BLOCK;
      case DXGI_FORMAT_BC7_UNORM_SRGB: return VK_FORMAT_BC7_SRGB_BLOCK;
      default: return VK_FORMAT_UNDEFINED;
      }
    }

    dxvk::TextureRef materialTextureFromDescriptor(
        const D3D12NativeBoundDescriptor& descriptor) {
      if (!g_device.ptr() || !g_rtxContext.ptr() || !descriptor.resource
       || descriptor.resourceDescription.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
       || !descriptor.capturedBytes)
        return {};

      const uint64_t key = descriptor.resource ^ (static_cast<uint64_t>(descriptor.format) << 48);
      const auto cached = g_nativeTextureRefs.find(key);
      if (cached != g_nativeTextureRefs.end())
        return cached->second;

      UINT snapshotSubresource = 0;
      if (descriptor.viewDescriptionSize >= sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC)) {
        const auto* srv = reinterpret_cast<const D3D12_SHADER_RESOURCE_VIEW_DESC*>(
          descriptor.viewDescription.data());
        const UINT mipLevels = descriptor.resourceDescription.MipLevels
          ? descriptor.resourceDescription.MipLevels : 1;
        switch (srv->ViewDimension) {
        case D3D12_SRV_DIMENSION_TEXTURE1D:
          snapshotSubresource = srv->Texture1D.MostDetailedMip;
          break;
        case D3D12_SRV_DIMENSION_TEXTURE1DARRAY:
          snapshotSubresource = srv->Texture1DArray.FirstArraySlice * mipLevels
            + srv->Texture1DArray.MostDetailedMip;
          break;
        case D3D12_SRV_DIMENSION_TEXTURE2D:
          snapshotSubresource = srv->Texture2D.MostDetailedMip;
          break;
        case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:
          snapshotSubresource = srv->Texture2DArray.FirstArraySlice * mipLevels
            + srv->Texture2DArray.MostDetailedMip;
          break;
        case D3D12_SRV_DIMENSION_TEXTURE3D:
          snapshotSubresource = srv->Texture3D.MostDetailedMip;
          break;
        case D3D12_SRV_DIMENSION_TEXTURECUBE:
          snapshotSubresource = srv->TextureCube.MostDetailedMip;
          break;
        case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY:
          snapshotSubresource = srv->TextureCubeArray.First2DArrayFace * mipLevels
            + srv->TextureCubeArray.MostDetailedMip;
          break;
        default:
          break;
        }
      }
      D3D12NativeTextureSnapshot snapshot;
      if (!getD3D12TextureSnapshot(descriptor.resource, snapshotSubresource, snapshot))
        return {};
      const VkFormat format = toVkMaterialFormat(snapshot.format != DXGI_FORMAT_UNKNOWN
        ? snapshot.format : descriptor.format);
      if (format == VK_FORMAT_UNDEFINED || !snapshot.width || !snapshot.height)
        return {};
      // Keep material creation out of the frame-critical path for large
      // streaming textures. Those resources remain on the existing default
      // material until an asynchronous uploader is available.
      if (snapshot.width > 128 || snapshot.height > 128
       || snapshot.bytes.size() > 64ull * 1024ull)
        return {};
      if (g_nativeTextureRefs.size() >= kMaxNativeTextureRefs
       || g_nativeTextureBytes > kMaxNativeTextureBytes - snapshot.bytes.size())
        return {};

      dxvk::DxvkImageCreateInfo imageInfo {};
      imageInfo.type = VK_IMAGE_TYPE_2D;
      imageInfo.format = format;
      imageInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
      imageInfo.extent = { snapshot.width, snapshot.height, 1 };
      imageInfo.numLayers = 1;
      imageInfo.mipLevels = 1;
      imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      imageInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT
        | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
        | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
      imageInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
      imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
      imageInfo.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

      auto image = g_device->createImage(
        imageInfo,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        dxvk::DxvkMemoryStats::Category::RTXMaterialTexture,
        "D3D12 material texture");
      if (!image.ptr())
        return {};

      const auto* formatInfo = image->formatInfo();
      const VkExtent3D extent {
        snapshot.width, snapshot.height, 1 };
      const VkDeviceSize rowPitch = snapshot.rowPitch
        ? snapshot.rowPitch : snapshot.rowSize;
      const VkDeviceSize layerPitch = rowPitch
        * std::max<uint32_t>(1, (snapshot.height + formatInfo->blockSize.height - 1)
          / formatInfo->blockSize.height);
      g_rtxContext->updateImage(
        image,
        VkImageSubresourceLayers { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        VkOffset3D { 0, 0, 0 },
        extent,
        snapshot.bytes.data(),
        rowPitch,
        layerPitch);

      dxvk::DxvkImageViewCreateInfo viewInfo {};
      viewInfo.type = VK_IMAGE_VIEW_TYPE_2D;
      viewInfo.format = format;
      viewInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
      viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
      viewInfo.numLevels = 1;
      viewInfo.numLayers = 1;
      auto view = g_device->createImageView(image, viewInfo);
      if (!view.ptr())
        return {};

      dxvk::TextureRef result(view);
      g_nativeTextureRefs.emplace(key, result);
      g_nativeTextureBytes += snapshot.bytes.size();
      return result;
    }

    dxvk::Rc<dxvk::DxvkSampler> materialSamplerFromDescriptors(
        const std::vector<D3D12NativeBoundDescriptor>& descriptors) {
      for (const auto& descriptor : descriptors) {
        if (descriptor.kind != D3D12NativeDescriptorKind::Sampler
         || descriptor.viewDescriptionSize < sizeof(D3D12_SAMPLER_DESC))
          continue;
        const auto* source = reinterpret_cast<const D3D12_SAMPLER_DESC*>(
          descriptor.viewDescription.data());
        const uint64_t samplerKey = XXH3_64bits(
          descriptor.viewDescription.data(), sizeof(D3D12_SAMPLER_DESC));
        const auto cached = g_nativeSamplers.find(samplerKey);
        if (cached != g_nativeSamplers.end())
          return cached->second;
        dxvk::DxvkSamplerCreateInfo samplerInfo {};
        // D3D12 packs min/mag/mip selection into the filter enum. Preserve
        // those choices instead of treating every material as trilinear.
        // Comparison filters add 0x80; masking that bit leaves the same
        // min/mag/mip pattern for the ordinary sampler fields.
        const uint32_t filter = static_cast<uint32_t>(source->Filter) & 0x7fu;
        const bool anisotropic = filter == 85u;
        const bool minLinear = anisotropic || filter == 16u || filter == 17u
          || filter == 20u || filter == 21u;
        const bool magLinear = anisotropic || filter == 4u || filter == 5u
          || filter == 20u || filter == 21u;
        const bool mipLinear = anisotropic || filter == 1u || filter == 5u
          || filter == 17u || filter == 21u;
        samplerInfo.magFilter = magLinear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        samplerInfo.minFilter = minLinear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        samplerInfo.mipmapMode = mipLinear
          ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.mipmapLodBias = source->MipLODBias;
        samplerInfo.mipmapLodMin = source->MinLOD;
        samplerInfo.mipmapLodMax = source->MaxLOD;
        samplerInfo.useAnisotropy = anisotropic && source->MaxAnisotropy > 1
          ? VK_TRUE : VK_FALSE;
        samplerInfo.maxAnisotropy = std::max<float>(
          1.0f, static_cast<float>(source->MaxAnisotropy));
        auto addressMode = [](D3D12_TEXTURE_ADDRESS_MODE mode) {
          switch (mode) {
          case D3D12_TEXTURE_ADDRESS_MODE_WRAP: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
          case D3D12_TEXTURE_ADDRESS_MODE_MIRROR: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
          case D3D12_TEXTURE_ADDRESS_MODE_BORDER: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
          case D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
          default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
          }
        };
        samplerInfo.addressModeU = addressMode(source->AddressU);
        samplerInfo.addressModeV = addressMode(source->AddressV);
        samplerInfo.addressModeW = addressMode(source->AddressW);
        samplerInfo.compareToDepth = (static_cast<uint32_t>(source->Filter) & 0x80u) != 0
          ? VK_TRUE : VK_FALSE;
        switch (source->ComparisonFunc) {
        case D3D12_COMPARISON_FUNC_NEVER: samplerInfo.compareOp = VK_COMPARE_OP_NEVER; break;
        case D3D12_COMPARISON_FUNC_LESS: samplerInfo.compareOp = VK_COMPARE_OP_LESS; break;
        case D3D12_COMPARISON_FUNC_EQUAL: samplerInfo.compareOp = VK_COMPARE_OP_EQUAL; break;
        case D3D12_COMPARISON_FUNC_LESS_EQUAL: samplerInfo.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL; break;
        case D3D12_COMPARISON_FUNC_GREATER: samplerInfo.compareOp = VK_COMPARE_OP_GREATER; break;
        case D3D12_COMPARISON_FUNC_NOT_EQUAL: samplerInfo.compareOp = VK_COMPARE_OP_NOT_EQUAL; break;
        case D3D12_COMPARISON_FUNC_GREATER_EQUAL: samplerInfo.compareOp = VK_COMPARE_OP_GREATER_OR_EQUAL; break;
        case D3D12_COMPARISON_FUNC_ALWAYS: samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS; break;
        default: samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS; break;
        }
        samplerInfo.borderColor = {
          { source->BorderColor[0], source->BorderColor[1],
            source->BorderColor[2], source->BorderColor[3] } };
        samplerInfo.usePixelCoord = VK_FALSE;
        auto sampler = g_device->createSampler(samplerInfo);
        g_nativeSamplers.emplace(samplerKey, sampler);
        return sampler;
      }
      return nullptr;
    }

    bool finiteCameraMatrix(const float* matrix) {
      for (size_t i = 0; i < 16; ++i) {
        if (!std::isfinite(matrix[i]) || std::abs(matrix[i]) > 1.0e8f)
          return false;
      }
      return true;
    }

    // A row-vector world-to-view matrix keeps three orthonormal rows and stores
    // the translation in its fourth row.
    bool buildWorldToView(const float* view, dxvk::Matrix4& worldToView) {
      if (!finiteCameraMatrix(view))
        return false;

      std::array<float, 9> rotation {};
      std::array<float, 3> rowLengths {};
      for (size_t row = 0; row < 3; ++row) {
        const float* r = view + row * 4;
        const float length = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
        if (std::abs(length - 1.0f) > 0.05f)
          return false;
        rowLengths[row] = length;
        for (size_t column = 0; column < 3; ++column)
          rotation[row * 3 + column] = r[column] / length;
      }

      const float dot01 = rotation[0] * rotation[3]
        + rotation[1] * rotation[4] + rotation[2] * rotation[5];
      const float dot02 = rotation[0] * rotation[6]
        + rotation[1] * rotation[7] + rotation[2] * rotation[8];
      const float dot12 = rotation[3] * rotation[6]
        + rotation[4] * rotation[7] + rotation[5] * rotation[8];
      if (std::abs(dot01) > 0.05f || std::abs(dot02) > 0.05f || std::abs(dot12) > 0.05f)
        return false;

      // Remix takes the camera position from the inverse of this matrix, so the
      // translation has to be rebuilt from the normalized rows. Keeping the
      // original one would place the eye at a distance scaled by the row length,
      // which reads as every camera translation being amplified. For a
      // row-vector matrix with an orthonormal rotation the eye is recovered by
      // eye_i = -sum_j t[j] * R[i][j], and the translation is its negation
      // projected back onto the rows.
      std::array<float, 3> eye {};
      for (size_t column = 0; column < 3; ++column) {
        eye[column] = -(view[12] * rotation[0 * 3 + column]
          + view[13] * rotation[1 * 3 + column]
          + view[14] * rotation[2 * 3 + column]);
      }
      std::array<float, 3> translation {};
      for (size_t column = 0; column < 3; ++column) {
        translation[column] = -(eye[0] * rotation[0 + column]
          + eye[1] * rotation[3 + column]
          + eye[2] * rotation[6 + column]);
      }

      static std::atomic<uint32_t> basisLogs { 0 };
      if (basisLogs.fetch_add(1, std::memory_order_relaxed) < 6) {
        log("D3D12 camera world-to-view: rowLengths=(%.6f,%.6f,%.6f) eye=(%.3f,%.3f,%.3f) t=(%.3f,%.3f,%.3f) tBuilt=(%.3f,%.3f,%.3f)",
          static_cast<double>(rowLengths[0]),
          static_cast<double>(rowLengths[1]),
          static_cast<double>(rowLengths[2]),
          static_cast<double>(eye[0]),
          static_cast<double>(eye[1]),
          static_cast<double>(eye[2]),
          static_cast<double>(view[12]),
          static_cast<double>(view[13]),
          static_cast<double>(view[14]),
          static_cast<double>(translation[0]),
          static_cast<double>(translation[1]),
          static_cast<double>(translation[2]));
      }

      worldToView = dxvk::Matrix4(
        rotation[0], rotation[1], rotation[2], 0.0f,
        rotation[3], rotation[4], rotation[5], 0.0f,
        rotation[6], rotation[7], rotation[8], 0.0f,
        translation[0], translation[1], translation[2], 1.0f);
      return true;
    }

    // Splits a row-vector world-to-projection matrix into a world-to-view and a
    // view-to-projection matrix so Remix can consume it.
    bool decomposeRowVectorViewProjection(
        const float* source,
        dxvk::Matrix4& worldToView,
        dxvk::Matrix4& viewToProjection,
        float& p00,
        float& p11,
        float& p22) {
      if (!finiteCameraMatrix(source))
        return false;

      const float forwardLength = std::sqrt(
        source[3] * source[3] + source[7] * source[7] + source[11] * source[11]);
      if (forwardLength < 0.8f || forwardLength > 1.2f)
        return false;

      p00 = std::sqrt(source[0] * source[0] + source[4] * source[4] + source[8] * source[8]);
      p11 = std::sqrt(source[1] * source[1] + source[5] * source[5] + source[9] * source[9]);
      if (p00 < 0.1f || p11 < 0.1f)
        return false;

      // A row-vector world-to-projection matrix keeps the view forward axis in
      // the last column, while the first two columns contain the X/Y projection
      // scales multiplied by the corresponding view axes.
      const float forwardX = source[3] / forwardLength;
      const float forwardY = source[7] / forwardLength;
      const float forwardZ = source[11] / forwardLength;
      p22 = source[2] * forwardX + source[6] * forwardY + source[10] * forwardZ;
      if (p22 < 0.1f || p22 > 10.0f)
        return false;

      std::array<float, 9> rotation {};
      for (size_t row = 0; row < 3; ++row) {
        rotation[row * 3 + 0] = source[row * 4 + 0] / p00;
        rotation[row * 3 + 1] = source[row * 4 + 1] / p11;
        rotation[row * 3 + 2] = source[row * 4 + 3] / forwardLength;

        const float rowLength = std::sqrt(
          rotation[row * 3 + 0] * rotation[row * 3 + 0]
          + rotation[row * 3 + 1] * rotation[row * 3 + 1]
          + rotation[row * 3 + 2] * rotation[row * 3 + 2]);
        if (std::abs(rowLength - 1.0f) > 0.08f
         || std::abs(rotation[row * 3 + 2] - source[row * 4 + 3]) > 0.08f)
          return false;
      }

      const float dot01 = rotation[0] * rotation[3]
        + rotation[1] * rotation[4] + rotation[2] * rotation[5];
      const float dot02 = rotation[0] * rotation[6]
        + rotation[1] * rotation[7] + rotation[2] * rotation[8];
      const float dot12 = rotation[3] * rotation[6]
        + rotation[4] * rotation[7] + rotation[5] * rotation[8];
      if (std::abs(dot01) > 0.08f || std::abs(dot02) > 0.08f || std::abs(dot12) > 0.08f)
        return false;

      const float tx = source[12] / p00;
      const float ty = source[13] / p11;
      const float tz = source[15];
      const float projectionB = source[14] - tz * p22;

      worldToView = dxvk::Matrix4(
        rotation[0], rotation[1], rotation[2], 0.0f,
        rotation[3], rotation[4], rotation[5], 0.0f,
        rotation[6], rotation[7], rotation[8], 0.0f,
        tx, ty, tz, 1.0f);
      viewToProjection = dxvk::Matrix4(
        p00, 0.0f, 0.0f, 0.0f,
        0.0f, -p11, 0.0f, 0.0f,
        0.0f, 0.0f, p22, 1.0f,
        0.0f, 0.0f, projectionB, 0.0f);
      return true;
    }

    uint32_t indexStride(DXGI_FORMAT format) {
      switch (format) {
      case DXGI_FORMAT_R16_UINT: return 2;
      case DXGI_FORMAT_R32_UINT: return 4;
      default: return 0;
      }
    }

    VkPrimitiveTopology toVkTopology(D3D12_PRIMITIVE_TOPOLOGY topology) {
      switch (topology) {
      case D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST:
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      case D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP:
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
      case D3D_PRIMITIVE_TOPOLOGY_LINELIST:
        return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
      case D3D_PRIMITIVE_TOPOLOGY_POINTLIST:
        return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
      default:
        return VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
      }
    }

    uint32_t readIndex(const std::vector<uint8_t>& data, uint32_t stride, size_t index) {
      const size_t offset = index * stride;
      if (offset + stride > data.size())
        return 0;
      if (stride == 2)
        return *reinterpret_cast<const uint16_t*>(data.data() + offset);
      return *reinterpret_cast<const uint32_t*>(data.data() + offset);
    }

    dxvk::Rc<dxvk::DxvkBuffer> createHostBuffer(const std::vector<uint8_t>& data, const char* name) {
      if (data.empty() || data.size() > kMaxNativeGeometryBytes)
        return nullptr;

      dxvk::DxvkBufferCreateInfo info;
      info.size = dxvk::align(data.size(), dxvk::CACHE_LINE_SIZE);
      info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        | VK_BUFFER_USAGE_TRANSFER_DST_BIT
        | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
        | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
        | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT
        | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT;

      auto buffer = g_device->createBuffer(
        info,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        dxvk::DxvkMemoryStats::Category::RTXAccelerationStructure,
        name);
      if (buffer == nullptr)
        return nullptr;

      dxvk::DxvkBufferSlice slice(buffer, 0, data.size());
      std::memcpy(slice.mapPtr(0), data.data(), data.size());
      return buffer;
    }

    uint64_t geometryKey(
        const std::vector<uint8_t>& vertexData,
        const std::vector<uint8_t>& indexData,
        uint32_t stride,
        uint32_t indexCount,
        VkPrimitiveTopology topology) {
      uint64_t hash = XXH3_64bits(vertexData.data(), vertexData.size());
      hash = XXH64(indexData.data(), indexData.size(), hash);
      hash = XXH64(&stride, sizeof(stride), hash);
      hash = XXH64(&indexCount, sizeof(indexCount), hash);
      hash = XXH64(&topology, sizeof(topology), hash);
      return hash;
    }

    struct CanonicalVertex {
      float position[3];
      float normal[3];
      float texcoord[2];
      uint32_t color;
    };

    // Draws whose transform lookup failed can still be placed when their
    // vertices are already in world space. Only the frontend's user knows that
    // for its engine, so placement is a policy instead of a guess.
    enum class WorldSpacePolicy : uint32_t {
      Reject = 0,
      ByExtent = 1,
      AcceptAll = 2,
    };

    std::atomic<uint32_t> g_worldSpacePolicy {
      static_cast<uint32_t>(WorldSpacePolicy::Reject) };
    std::atomic<float> g_worldSpaceMinExtent { 100.0f };

    bool shouldPlaceAsWorldSpace(const dxvk::AxisAlignedBoundingBox& bounds) {
      const auto policy = static_cast<WorldSpacePolicy>(
        g_worldSpacePolicy.load(std::memory_order_relaxed));
      if (policy == WorldSpacePolicy::Reject || !bounds.isValid())
        return false;
      if (policy == WorldSpacePolicy::AcceptAll)
        return true;

      const float extent = std::max(
        bounds.maxPos.x - bounds.minPos.x,
        std::max(bounds.maxPos.y - bounds.minPos.y,
          bounds.maxPos.z - bounds.minPos.z));
      return extent >= g_worldSpaceMinExtent.load(std::memory_order_relaxed);
    }

    bool isIdentityTransform(const std::array<float, 16>& transform) {
      return std::abs(transform[0] - 1.0f) < 0.001f
        && std::abs(transform[5] - 1.0f) < 0.001f
        && std::abs(transform[10] - 1.0f) < 0.001f
        && std::abs(transform[15] - 1.0f) < 0.001f
        && std::abs(transform[1]) < 0.001f
        && std::abs(transform[2]) < 0.001f
        && std::abs(transform[4]) < 0.001f
        && std::abs(transform[6]) < 0.001f
        && std::abs(transform[8]) < 0.001f
        && std::abs(transform[9]) < 0.001f
        && std::abs(transform[12]) < 0.001f
        && std::abs(transform[13]) < 0.001f
        && std::abs(transform[14]) < 0.001f;
    }

    uint32_t formatSize(DXGI_FORMAT format) {
      switch (format) {
      case DXGI_FORMAT_R32_FLOAT: return 4;
      case DXGI_FORMAT_R32G32_FLOAT: return 8;
      case DXGI_FORMAT_R32G32B32_FLOAT: return 12;
      case DXGI_FORMAT_R32G32B32A32_FLOAT: return 16;
      case DXGI_FORMAT_R16G16_FLOAT: return 4;
      case DXGI_FORMAT_R16G16B16A16_FLOAT: return 8;
      case DXGI_FORMAT_R8G8B8A8_UNORM:
      case DXGI_FORMAT_B8G8R8A8_UNORM: return 4;
      default: return 0;
      }
    }

    bool readFloatAttribute(
        const uint8_t* vertex,
        size_t available,
        DXGI_FORMAT format,
        float* output,
        uint32_t components) {
      const uint32_t bytes = formatSize(format);
      if (!bytes || bytes > available || !output)
        return false;
      if (format == DXGI_FORMAT_R32_FLOAT || format == DXGI_FORMAT_R32G32_FLOAT
       || format == DXGI_FORMAT_R32G32B32_FLOAT || format == DXGI_FORMAT_R32G32B32A32_FLOAT) {
        const uint32_t sourceComponents = bytes / sizeof(float);
        const auto* values = reinterpret_cast<const float*>(vertex);
        for (uint32_t i = 0; i < components; ++i)
          output[i] = i < sourceComponents ? values[i] : (i == 3 ? 1.0f : 0.0f);
        return true;
      }
      if (format == DXGI_FORMAT_R16G16_FLOAT
       || format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
        const uint32_t sourceComponents = bytes / sizeof(uint16_t);
        const auto* values = reinterpret_cast<const uint16_t*>(vertex);
        auto halfToFloat = [](uint16_t value) {
          const uint32_t sign = (static_cast<uint32_t>(value & 0x8000u)) << 16;
          const uint32_t exponent = (value >> 10) & 0x1fu;
          const uint32_t mantissa = value & 0x3ffu;
          uint32_t bits = 0;
          if (exponent == 0) {
            if (!mantissa) {
              bits = sign;
            } else {
              uint32_t normalized = mantissa;
              uint32_t shift = 0;
              while ((normalized & 0x400u) == 0) {
                normalized <<= 1;
                ++shift;
              }
              bits = sign
                | ((static_cast<uint32_t>(127 - 15 - shift + 1)) << 23)
                | ((normalized & 0x3ffu) << 13);
            }
          } else if (exponent == 0x1fu) {
            bits = sign | 0x7f800000u | (mantissa << 13);
          } else {
            bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
          }
          float result = 0.0f;
          std::memcpy(&result, &bits, sizeof(result));
          return result;
        };
        for (uint32_t i = 0; i < components; ++i)
          output[i] = i < sourceComponents ? halfToFloat(values[i])
            : (i == 3 ? 1.0f : 0.0f);
        return true;
      }
      return false;
    }

    bool readColorAttribute(
        const uint8_t* vertex,
        size_t available,
        DXGI_FORMAT format,
        uint32_t& output) {
      if (format != DXGI_FORMAT_R8G8B8A8_UNORM && format != DXGI_FORMAT_B8G8R8A8_UNORM)
        return false;
      if (available < 4)
        return false;
      std::memcpy(&output, vertex, sizeof(output));
      if (format == DXGI_FORMAT_R8G8B8A8_UNORM) {
        const uint8_t r = (output >> 0) & 0xff;
        const uint8_t b = (output >> 16) & 0xff;
        output = (output & 0xff00ff00u) | (static_cast<uint32_t>(b) << 0) | (static_cast<uint32_t>(r) << 16);
      }
      return true;
    }

    bool convertVertexLayout(
        const std::vector<uint8_t>& source,
        uint32_t sourceStride,
        uint32_t vertexCount,
        const D3D12NativePipelineInfo* pipelineInfo,
        std::vector<uint8_t>& output) {
      if (!sourceStride || vertexCount == 0
       || source.size() < static_cast<size_t>(sourceStride) * vertexCount)
        return false;

      const D3D12NativeInputElement* position = nullptr;
      const D3D12NativeInputElement* normal = nullptr;
      const D3D12NativeInputElement* texcoord = nullptr;
      const D3D12NativeInputElement* color = nullptr;
      if (pipelineInfo) {
        for (const auto& element : pipelineInfo->inputElements) {
          if (element.slot != 0)
            continue;
          std::string semantic = element.semantic;
          std::transform(semantic.begin(), semantic.end(), semantic.begin(),
            [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
          if (semantic == "POSITION" && !position)
            position = &element;
          else if (semantic == "NORMAL" && !normal)
            normal = &element;
          else if (semantic == "TEXCOORD" && element.semanticIndex == 0 && !texcoord)
            texcoord = &element;
          else if (semantic == "COLOR" && element.semanticIndex == 0 && !color)
            color = &element;
        }
      }

      if (!position && pipelineInfo) {
        // Model draws name their attributes ATTRIBUTE0..3 rather than POSITION and
        // keep the position in the first one, so the lowest slot 0 offset stands in
        // for it. Without this every such vertex collapses to the origin.
        for (const auto& element : pipelineInfo->inputElements) {
          if (element.slot != 0 || element.offset == D3D12_APPEND_ALIGNED_ELEMENT)
            continue;
          if (!position || element.offset < position->offset)
            position = &element;
        }
      }

      if (position) {
        const DXGI_FORMAT format = position->format;
        const bool supported = format == DXGI_FORMAT_R32_FLOAT
          || format == DXGI_FORMAT_R32G32_FLOAT
          || format == DXGI_FORMAT_R32G32B32_FLOAT
          || format == DXGI_FORMAT_R32G32B32A32_FLOAT
          || format == DXGI_FORMAT_R16G16_FLOAT
          || format == DXGI_FORMAT_R16G16B16A16_FLOAT;
        if (!supported) {
          static std::mutex formatMutex;
          static std::unordered_set<uint32_t> loggedFormats;
          std::lock_guard lock(formatMutex);
          if (loggedFormats.insert(static_cast<uint32_t>(format)).second) {
            log("D3D12 unsupported position format: format=%u vs=0x%016llx stride=%u offset=%u",
              static_cast<unsigned>(format),
              static_cast<unsigned long long>(pipelineInfo->vertexShaderHash),
              pipelineInfo->vertexBufferStride, position->offset);
          }
        }
      }

      output.resize(static_cast<size_t>(vertexCount) * sizeof(CanonicalVertex));
      for (uint32_t i = 0; i < vertexCount; ++i) {
        const auto* sourceVertex = source.data() + static_cast<size_t>(i) * sourceStride;
        CanonicalVertex converted {};
        converted.normal[2] = 1.0f;
        converted.color = 0xffffffffu;
        auto readAt = [&](const D3D12NativeInputElement* element, float* dst, uint32_t count) {
          return element && element->offset != D3D12_APPEND_ALIGNED_ELEMENT
            && element->offset < sourceStride
            && readFloatAttribute(sourceVertex + element->offset,
              sourceStride - element->offset, element->format, dst, count);
        };
        if (!readAt(position, converted.position, 3)) {
          if (!readFloatAttribute(sourceVertex, sourceStride, DXGI_FORMAT_R32G32B32_FLOAT,
              converted.position, 3))
            return false;
        }
        if (normal)
          readAt(normal, converted.normal, 3);
        if (texcoord)
          readAt(texcoord, converted.texcoord, 2);
        if (color && color->offset != D3D12_APPEND_ALIGNED_ELEMENT
         && color->offset < sourceStride)
          readColorAttribute(sourceVertex + color->offset, sourceStride - color->offset,
            color->format, converted.color);
        std::memcpy(output.data() + static_cast<size_t>(i) * sizeof(CanonicalVertex),
          &converted, sizeof(converted));
      }
      return true;
    }

  namespace {

    // A capture stage that rejects a draw has to say so once, otherwise the draw
    // simply disappears from the image without any trace.
    void logBridgeStageOnce(const char* stage) {
      static std::mutex stageMutex;
      static std::unordered_set<std::string> loggedStages;
      std::lock_guard lock(stageMutex);
      if (loggedStages.insert(stage).second)
        log("D3D12 geometry bridge stage rejected: %s", stage);
    }

    // Bounds of the transforms submitted this frame. They frame the scene when the
    // engine's packed camera block cannot be decoded into a standard view.
    std::mutex g_boundsMutex;
    float g_boundsMin[3] = { 0.0f, 0.0f, 0.0f };
    float g_boundsMax[3] = { 0.0f, 0.0f, 0.0f };
    uint32_t g_boundsCount = 0;

    void accumulateInstanceBounds(const D3D12NativePipelineInfo& info) {
      if (!info.hasObjectToWorld)
        return;

      const float x = info.objectToWorld[12];
      const float y = info.objectToWorld[13];
      const float z = info.objectToWorld[14];
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        return;

      std::lock_guard lock(g_boundsMutex);
      if (g_boundsCount == 0) {
        g_boundsMin[0] = g_boundsMax[0] = x;
        g_boundsMin[1] = g_boundsMax[1] = y;
        g_boundsMin[2] = g_boundsMax[2] = z;
      } else {
        g_boundsMin[0] = std::min(g_boundsMin[0], x);
        g_boundsMax[0] = std::max(g_boundsMax[0], x);
        g_boundsMin[1] = std::min(g_boundsMin[1], y);
        g_boundsMax[1] = std::max(g_boundsMax[1], y);
        g_boundsMin[2] = std::min(g_boundsMin[2], z);
        g_boundsMax[2] = std::max(g_boundsMax[2], z);
      }
      ++g_boundsCount;
    }

  }

    bool commitNativeMeshInstance(
        remixapi_MeshHandle handle,
        const D3D12NativePipelineInfo* pipelineInfo,
        bool worldSpaceGeometry = false) {
      // An unresolved object transform is not a safe fallback: using an identity
      // matrix would pile unrelated parts at the world origin.
      if (!handle || !g_sceneInitialized || !g_rtxContext.ptr() || !pipelineInfo) {
        logBridgeStageOnce("instance not ready");
        g_nativeNoTransform.fetch_add(1, std::memory_order_relaxed);
        noteAircraftDraw(pipelineInfo ? pipelineInfo->vertexShaderHash : 0ull,
          kAircraftNoTransform);
        const uint32_t diagnostic = g_transformRejectionDiagnostics.fetch_add(
          1, std::memory_order_relaxed);
        if (diagnostic < 64 && pipelineInfo) {
          log("D3D12 rejected draw: reason=no-transform draw=%llu frame=%u vs=0x%016llx ps=0x%016llx vb=0x%llx/%u/%u ib=0x%llx/%u/%u args=(%u,%u,%u,%d)",
            static_cast<unsigned long long>(pipelineInfo->drawId),
            g_device.ptr() ? g_device->getCurrentFrameId() : 0u,
            static_cast<unsigned long long>(pipelineInfo->vertexShaderHash),
            static_cast<unsigned long long>(pipelineInfo->pixelShaderHash),
            static_cast<unsigned long long>(pipelineInfo->vertexBufferLocation),
            pipelineInfo->vertexBufferSize, pipelineInfo->vertexBufferStride,
            static_cast<unsigned long long>(pipelineInfo->indexBufferLocation),
            pipelineInfo->indexBufferSize, static_cast<unsigned>(pipelineInfo->indexBufferFormat),
            pipelineInfo->indexCount, pipelineInfo->instanceCount,
            pipelineInfo->startIndexLocation, pipelineInfo->baseVertexLocation);
        }
        return false;
      }

      // Geometry authored directly in world space carries an identity
      // transform, so an identity matrix here is a valid placement rather than
      // a failed lookup. The caller decides which draws that applies to; see
      // shouldPlaceAsWorldSpace().
      D3D12NativePipelineInfo worldSpaceInfo;
      const bool worldSpaceCandidate = worldSpaceGeometry;

      if (!pipelineInfo->hasObjectToWorld) {
        if (!worldSpaceCandidate) {
          logBridgeStageOnce("instance no transform");
          g_nativeNoTransform.fetch_add(1, std::memory_order_relaxed);
          noteAircraftDraw(pipelineInfo->vertexShaderHash, kAircraftNoTransform);
          const uint32_t diagnostic = g_transformRejectionDiagnostics.fetch_add(
            1, std::memory_order_relaxed);
          if (diagnostic < 64) {
            log("D3D12 rejected draw: reason=no-transform draw=%llu frame=%u vs=0x%016llx ps=0x%016llx vb=0x%llx/%u/%u ib=0x%llx/%u/%u args=(%u,%u,%u,%d) tables=(0x%llx,0x%llx,0x%llx,0x%llx)",
              static_cast<unsigned long long>(pipelineInfo->drawId),
              g_device.ptr() ? g_device->getCurrentFrameId() : 0u,
              static_cast<unsigned long long>(pipelineInfo->vertexShaderHash),
              static_cast<unsigned long long>(pipelineInfo->pixelShaderHash),
              static_cast<unsigned long long>(pipelineInfo->vertexBufferLocation),
              pipelineInfo->vertexBufferSize, pipelineInfo->vertexBufferStride,
              static_cast<unsigned long long>(pipelineInfo->indexBufferLocation),
              pipelineInfo->indexBufferSize, static_cast<unsigned>(pipelineInfo->indexBufferFormat),
              pipelineInfo->indexCount, pipelineInfo->instanceCount,
              pipelineInfo->startIndexLocation, pipelineInfo->baseVertexLocation,
              static_cast<unsigned long long>(pipelineInfo->descriptorTables[0]),
              static_cast<unsigned long long>(pipelineInfo->descriptorTables[1]),
              static_cast<unsigned long long>(pipelineInfo->descriptorTables[2]),
              static_cast<unsigned long long>(pipelineInfo->descriptorTables[3]));
          }
          return false;
        }
        worldSpaceInfo = *pipelineInfo;
        worldSpaceInfo.hasObjectToWorld = true;
        worldSpaceInfo.objectTransformSource = 3;
        worldSpaceInfo.objectToWorld = {};
        worldSpaceInfo.objectToWorld[0] = 1.0f;
        worldSpaceInfo.objectToWorld[5] = 1.0f;
        worldSpaceInfo.objectToWorld[10] = 1.0f;
        worldSpaceInfo.objectToWorld[15] = 1.0f;
        pipelineInfo = &worldSpaceInfo;
        g_nativeWorldSpaceFallbacks.fetch_add(1, std::memory_order_relaxed);
      }

      // An identity matrix is a valid placement only for geometry whose
      // vertices are already in world space. Anything else would stack the
      // mesh at the world origin, so such a draw is dropped instead.
      const auto& transform = pipelineInfo->objectToWorld;
      const bool identityTransform = isIdentityTransform(transform);

      // An identity object matrix means the vertices are in model space and the
      // placement lives in the draw's fused matrix at offset 0x000, which is not
      // decoded yet; see acz-model-transform.md. Such draws are dropped rather
      // than piled at the world origin.
      constexpr bool kAcceptIdentityAsWorldSpace = false;
      if (identityTransform && !worldSpaceCandidate && !kAcceptIdentityAsWorldSpace) {
        logBridgeStageOnce("instance identity");
        g_nativeIdentityFiltered.fetch_add(1, std::memory_order_relaxed);
        noteAircraftDraw(pipelineInfo->vertexShaderHash, kAircraftIdentity);
        const uint32_t filtered = g_filteredOriginDrawCount.fetch_add(
          1, std::memory_order_relaxed);
        if (filtered < 64) {
          log("D3D12 rejected draw: reason=identity draw=%llu frame=%u source=%u vs=0x%016llx ps=0x%016llx vb=0x%llx/%u/%u ib=0x%llx/%u/%u args=(%u,%u,%u,%d)",
            static_cast<unsigned long long>(pipelineInfo->drawId),
            g_device->getCurrentFrameId(),
            static_cast<unsigned>(pipelineInfo->objectTransformSource),
            static_cast<unsigned long long>(pipelineInfo->vertexShaderHash),
            static_cast<unsigned long long>(pipelineInfo->pixelShaderHash),
            static_cast<unsigned long long>(pipelineInfo->vertexBufferLocation),
            pipelineInfo->vertexBufferSize, pipelineInfo->vertexBufferStride,
            static_cast<unsigned long long>(pipelineInfo->indexBufferLocation),
            pipelineInfo->indexBufferSize, static_cast<unsigned>(pipelineInfo->indexBufferFormat),
            pipelineInfo->indexCount, pipelineInfo->instanceCount,
            pipelineInfo->startIndexLocation, pipelineInfo->baseVertexLocation);
        }
        return false;
      }
      if (identityTransform && worldSpaceCandidate
       && pipelineInfo->objectTransformSource != 3) {
        worldSpaceInfo = *pipelineInfo;
        worldSpaceInfo.objectTransformSource = 3;
        pipelineInfo = &worldSpaceInfo;
        g_nativeWorldSpaceFallbacks.fetch_add(1, std::memory_order_relaxed);
      }

      auto& sceneManager = g_device->getCommon()->getSceneManager();
      const uint32_t instancesBefore = sceneManager.getInstanceManager().getActiveCount();
      auto state = std::make_unique<dxvk::ExternalDrawState>();
      state->mesh = handle;
      state->cameraType = dxvk::CameraType::Main;
      if (pipelineInfo && pipelineInfo->hasObjectToWorld) {
        const auto& m = pipelineInfo->objectToWorld;
        const uint32_t diagnosticIndex = g_transformDiagnosticCount.fetch_add(1, std::memory_order_relaxed);
        {
          // Distance from the camera eye tells whether the object the scene is
          // framed around is among the submitted instances at all.
          const float dx = m[12] - g_cameraEye.x;
          const float dy = m[13] - g_cameraEye.y;
          const float dz = m[14] - g_cameraEye.z;
          const float distanceSq = dx * dx + dy * dy + dz * dz;
          float previous = g_nativeNearestInstanceDistanceSq.load(std::memory_order_relaxed);
          while (distanceSq < previous
              && !g_nativeNearestInstanceDistanceSq.compare_exchange_weak(
                   previous, distanceSq, std::memory_order_relaxed)) {
          }
          if (distanceSq < 1.0e6f)
            g_nativeNearInstances.fetch_add(1, std::memory_order_relaxed);
        }
        if (diagnosticIndex < 512) {
          const float translationLength = std::sqrt(
            m[12] * m[12] + m[13] * m[13] + m[14] * m[14]);
          log("D3D12 native Remix object transform: sample=%u frame=%u mesh=%p t=(%.6g,%.6g,%.6g) len=%.6g r0=(%.6g,%.6g,%.6g) r1=(%.6g,%.6g,%.6g) r2=(%.6g,%.6g,%.6g)",
            diagnosticIndex, g_device->getCurrentFrameId(), handle,
            static_cast<double>(m[12]), static_cast<double>(m[13]), static_cast<double>(m[14]),
            static_cast<double>(translationLength),
            static_cast<double>(m[0]), static_cast<double>(m[1]), static_cast<double>(m[2]),
            static_cast<double>(m[4]), static_cast<double>(m[5]), static_cast<double>(m[6]),
            static_cast<double>(m[8]), static_cast<double>(m[9]), static_cast<double>(m[10]));
        }
        state->drawCall.modifyTransformData().objectToWorld = dxvk::Matrix4(
          m[0], m[1], m[2], m[3],
          m[4], m[5], m[6], m[7],
          m[8], m[9], m[10], m[11],
          m[12], m[13], m[14], m[15]);
      } else {
        state->drawCall.modifyTransformData().objectToWorld = dxvk::Matrix4();
      }
      state->drawCall.modifyCategoryFlags().clrAll();
      state->doubleSided = true;
      const uint32_t accepted = g_nativeRemixApi.commitExternalGeometry(std::move(state));
      if (accepted) {
        g_nativeAcceptedInstances.fetch_add(1, std::memory_order_relaxed);
        noteAircraftDraw(pipelineInfo->vertexShaderHash, kAircraftSubmitted);
      accumulateInstanceBounds(*pipelineInfo);
      {
        // Model instances are sampled across the whole session rather than at the
        // start, so the frames the game is actually played in are covered. The
        // distance to the camera says whether the instance is where it belongs.
        static std::atomic<uint32_t> instanceLogs { 0 };
        const uint32_t instanceLog = instanceLogs.fetch_add(1, std::memory_order_relaxed);
        if ((instanceLog % 256) == 0 && instanceLog < 8192) {
          const auto& m = pipelineInfo->objectToWorld;
          const float dx = m[12] - g_cameraEye.x;
          const float dy = m[13] - g_cameraEye.y;
          const float dz = m[14] - g_cameraEye.z;
          log("ACZ instance: draw=%llu vs=0x%016llx t=(%.2f,%.2f,%.2f) eyeDistance=%.1f indices=%u stride=%u",
            static_cast<unsigned long long>(pipelineInfo->drawId),
            static_cast<unsigned long long>(pipelineInfo->vertexShaderHash),
            static_cast<double>(m[12]), static_cast<double>(m[13]),
            static_cast<double>(m[14]),
            static_cast<double>(std::sqrt(dx * dx + dy * dy + dz * dz)),
            pipelineInfo->indexCount, pipelineInfo->vertexBufferStride);
        }
      }
        if (worldSpaceCandidate)
          g_nativeWorldSpaceInstances.fetch_add(1, std::memory_order_relaxed);
      } else {
        noteAircraftDraw(pipelineInfo->vertexShaderHash, kAircraftApiRejected);
      }
      // The player's aircraft is whichever airframe lands nearest the camera, so
      // one line per frame reports that one instead of sampling whichever draw
      // the counter happens to hit. The frame that just ended is flushed when the
      // next one starts.
      if (pipelineInfo->vertexShaderHash == 0xb6a7d1d096ce8fe7ull
       && pipelineInfo->indexCount >= 512) {
        struct NearestAirframe {
          uint32_t frame = 0;
          uint64_t draw = 0;
          float distance = 0.0f;
          float position[3] = {};
          uint32_t indices = 0;
          uint32_t accepted = 0;
        };
        static NearestAirframe nearest;
        const uint32_t frame = g_device.ptr() ? g_device->getCurrentFrameId() : 0u;
        if (frame != nearest.frame) {
          if (nearest.draw) {
            log("ACZ nearest airframe: frame=%u draw=%llu accepted=%u t=(%.2f,%.2f,%.2f) eye=(%.1f,%.1f,%.1f) eyeDistance=%.1f indices=%u",
              nearest.frame,
              static_cast<unsigned long long>(nearest.draw),
              nearest.accepted,
              static_cast<double>(nearest.position[0]),
              static_cast<double>(nearest.position[1]),
              static_cast<double>(nearest.position[2]),
              static_cast<double>(g_cameraEye.x), static_cast<double>(g_cameraEye.y),
              static_cast<double>(g_cameraEye.z),
              static_cast<double>(nearest.distance),
              nearest.indices);
          }
          nearest = {};
          nearest.frame = frame;
        }
        const auto& m = pipelineInfo->objectToWorld;
        const float dx = m[12] - g_cameraEye.x;
        const float dy = m[13] - g_cameraEye.y;
        const float dz = m[14] - g_cameraEye.z;
        const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (!nearest.draw || distance < nearest.distance) {
          nearest.draw = pipelineInfo->drawId;
          nearest.distance = distance;
          nearest.position[0] = m[12];
          nearest.position[1] = m[13];
          nearest.position[2] = m[14];
          nearest.indices = pipelineInfo->indexCount;
          nearest.accepted = accepted ? 1u : 0u;
        }

        // Whether the airframe's vertices are already in world space decides if it
        // needs a placement at all, so the first two are logged beside the matrix
        // the layer would apply to them.
        if (distance > 1000.0f && pipelineInfo->vertexBufferLocation
         && pipelineInfo->vertexBufferStride >= sizeof(float) * 3) {
          static std::atomic<uint32_t> vertexLogs { 0 };
          if (vertexLogs.fetch_add(1, std::memory_order_relaxed) < 6) {
            std::vector<uint8_t> bytes;
            if (readGpuBuffer(pipelineInfo->vertexBufferLocation,
                  static_cast<size_t>(pipelineInfo->vertexBufferStride) * 2, bytes)
             && bytes.size() >= sizeof(float) * 6) {
              const auto* v0 = reinterpret_cast<const float*>(bytes.data());
              const auto* v1 = reinterpret_cast<const float*>(
                bytes.data() + pipelineInfo->vertexBufferStride);
              log("ACZ airframe vertices: draw=%llu stride=%u v0=(%.2f,%.2f,%.2f) v1=(%.2f,%.2f,%.2f) appliedT=(%.2f,%.2f,%.2f)",
                static_cast<unsigned long long>(pipelineInfo->drawId),
                pipelineInfo->vertexBufferStride,
                static_cast<double>(v0[0]), static_cast<double>(v0[1]),
                static_cast<double>(v0[2]),
                static_cast<double>(v1[0]), static_cast<double>(v1[1]),
                static_cast<double>(v1[2]),
                static_cast<double>(pipelineInfo->objectToWorld[12]),
                static_cast<double>(pipelineInfo->objectToWorld[13]),
                static_cast<double>(pipelineInfo->objectToWorld[14]));
            }
          }
        }

        // The airframe whose placement is wrong gets its whole candidate block
        // dumped, so the offset that holds the world position is found by content
        // instead of by assumption.
        if (distance > 1000.0f && pipelineInfo->objectTransformCbv) {
          static std::atomic<uint32_t> blockDumps { 0 };
          static uint32_t lastBlockFrame = 0;
          if (blockDumps.fetch_add(1, std::memory_order_relaxed) < 3
           && frame != lastBlockFrame) {
            lastBlockFrame = frame;
            std::vector<uint8_t> block;
            if (readGpuBuffer(pipelineInfo->objectTransformCbv, 0x1A0, block)
             && block.size() >= 0x1A0) {
              log("ACZ airframe block: frame=%u draw=%llu cbv=0x%llx eye=(%.1f,%.1f,%.1f)",
                frame, static_cast<unsigned long long>(pipelineInfo->drawId),
                static_cast<unsigned long long>(pipelineInfo->objectTransformCbv),
                static_cast<double>(g_cameraEye.x), static_cast<double>(g_cameraEye.y),
                static_cast<double>(g_cameraEye.z));
              for (uint32_t row = 0; row < 26; ++row) {
                const auto* v = reinterpret_cast<const float*>(block.data() + row * 16);
                log("ACZ airframe block +0x%03X: %13.4f %13.4f %13.4f %13.4f",
                  row * 16,
                  static_cast<double>(v[0]), static_cast<double>(v[1]),
                  static_cast<double>(v[2]), static_cast<double>(v[3]));
              }
            }
          }
        }
      }
      const bool instanceAdded =
        accepted != 0 && sceneManager.getInstanceManager().getActiveCount() > instancesBefore;
      logBridgeStageOnce(instanceAdded ? "instance accepted" : "instance not added");
      return instanceAdded;
    }

    bool submitNativeMesh(
        const std::vector<uint8_t>& vertexData,
        const std::vector<uint8_t>& indexData,
        uint32_t vertexStride,
        uint32_t vertexCount,
        uint32_t indexCount,
        VkIndexType indexType,
        VkPrimitiveTopology topology,
        uint64_t key,
        const D3D12NativePipelineInfo* pipelineInfo) {
      if (!g_sceneInitialized || !g_rtxContext.ptr() || vertexData.empty() || indexData.empty()
       || vertexStride < sizeof(float) * 3 || vertexCount == 0 || indexCount < 3
       || topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST) {
        static std::mutex gateMutex;
        static bool gateLogged = false;
        noteAircraftDraw(pipelineInfo ? pipelineInfo->vertexShaderHash : 0ull,
          kAircraftGated);
        std::lock_guard lock(gateMutex);
        if (!gateLogged) {
          gateLogged = true;
          log("D3D12 native mesh submit gate: sceneInit=%u context=%u vertexBytes=%zu indexBytes=%zu stride=%u vertices=%u indices=%u topology=%u",
            g_sceneInitialized ? 1u : 0u, g_rtxContext.ptr() ? 1u : 0u,
            vertexData.size(), indexData.size(), vertexStride, vertexCount,
            indexCount, static_cast<unsigned>(topology));
        }
        return false;
      }

      auto cachedMesh = g_nativeMeshHandles.find(key);
      if (cachedMesh != g_nativeMeshHandles.end()) {
        const bool accepted = commitNativeMeshInstance(cachedMesh->second, pipelineInfo,
          g_nativeWorldSpaceMeshes.find(key) != g_nativeWorldSpaceMeshes.end());
        logBridgeStageOnce(accepted ? "cached instance accepted" : "cached instance rejected");
        return accepted;
      }

      std::vector<uint8_t> canonicalVertexData;
      if (!convertVertexLayout(vertexData, vertexStride, vertexCount, pipelineInfo, canonicalVertexData)) {
        g_nativeConversionFailures.fetch_add(1, std::memory_order_relaxed);
        logBridgeStageOnce("vertex layout");
        return false;
      }
      constexpr uint32_t canonicalStride = sizeof(CanonicalVertex);
      auto vertexBuffer = createHostBuffer(canonicalVertexData, "D3D12 native vertex buffer");
      auto indexBuffer = createHostBuffer(indexData, "D3D12 native index buffer");
      if (!vertexBuffer.ptr() || !indexBuffer.ptr()) {
        logBridgeStageOnce("host buffer");
        return false;
      }

      dxvk::RasterGeometry geometry;
      geometry.vertexCount = vertexCount;
      geometry.indexCount = indexCount;
      // The parameter is already a Vulkan topology and the caller only reaches
      // this point with a triangle topology, so it is stored as it arrives.
      geometry.topology = topology;
      geometry.cullMode = VK_CULL_MODE_NONE;
      geometry.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

      const dxvk::DxvkBufferSlice vertexSlice(vertexBuffer, 0, canonicalVertexData.size());
      const dxvk::DxvkBufferSlice indexSlice(indexBuffer, 0, indexData.size());
      geometry.positionBuffer = dxvk::RasterBuffer(
        vertexSlice, offsetof(CanonicalVertex, position), canonicalStride, VK_FORMAT_R32G32B32_SFLOAT);
      geometry.normalBuffer = dxvk::RasterBuffer(
        vertexSlice, offsetof(CanonicalVertex, normal), canonicalStride, VK_FORMAT_R32G32B32_SFLOAT);
      geometry.texcoordBuffer = dxvk::RasterBuffer(
        vertexSlice, offsetof(CanonicalVertex, texcoord), canonicalStride, VK_FORMAT_R32G32_SFLOAT);
      geometry.color0Buffer = dxvk::RasterBuffer(
        vertexSlice, offsetof(CanonicalVertex, color), canonicalStride, VK_FORMAT_B8G8R8A8_UNORM);
      geometry.indexBuffer = dxvk::RasterBuffer(
        indexSlice, 0, sizeof(uint32_t), indexType);

      geometry.hashes[dxvk::HashComponents::VertexPosition] =
        XXH3_64bits(canonicalVertexData.data(), canonicalVertexData.size());
      geometry.hashes[dxvk::HashComponents::Indices] =
        XXH3_64bits(indexData.data(), indexData.size());
      geometry.hashes[dxvk::HashComponents::VertexLayout] =
        XXH64(&canonicalStride, sizeof(canonicalStride), geometry.hashes[dxvk::HashComponents::VertexPosition]);
      geometry.hashes[dxvk::HashComponents::GeometryDescriptor] =
        XXH64(&topology, sizeof(topology), geometry.hashes[dxvk::HashComponents::Indices]);
      geometry.hashes.precombine();

      dxvk::AxisAlignedBoundingBox bounds;
      for (uint32_t i = 0; i < vertexCount; ++i) {
        const auto* position = reinterpret_cast<const CanonicalVertex*>(canonicalVertexData.data() + static_cast<size_t>(i) * canonicalStride)->position;
        if (static_cast<size_t>(i) * canonicalStride + sizeof(CanonicalVertex) > canonicalVertexData.size())
          break;
        bounds.minPos.x = std::min(bounds.minPos.x, position[0]);
        bounds.minPos.y = std::min(bounds.minPos.y, position[1]);
        bounds.minPos.z = std::min(bounds.minPos.z, position[2]);
        bounds.maxPos.x = std::max(bounds.maxPos.x, position[0]);
        bounds.maxPos.y = std::max(bounds.maxPos.y, position[1]);
        bounds.maxPos.z = std::max(bounds.maxPos.z, position[2]);
      }
      geometry.boundingBox = bounds;
      const bool worldSpaceGeometry = shouldPlaceAsWorldSpace(bounds);

      remixapi_MaterialHandle materialHandle = nullptr;
      if (pipelineInfo) {
        uint64_t materialKey = pipelineInfo->pixelShaderHash;
        materialKey = XXH64(&pipelineInfo->vertexShaderHash, sizeof(pipelineInfo->vertexShaderHash), materialKey);
        materialKey = XXH64(&pipelineInfo->blendEnabled, sizeof(pipelineInfo->blendEnabled), materialKey);
        materialKey = XXH64(&pipelineInfo->alphaTestEnabled, sizeof(pipelineInfo->alphaTestEnabled), materialKey);
        const D3D12NativeBoundDescriptor* albedoDescriptor = nullptr;
        for (const auto& descriptor : pipelineInfo->boundDescriptors) {
          if (descriptor.kind == D3D12NativeDescriptorKind::Srv
           && descriptor.resourceDescription.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
           && descriptor.capturedBytes) {
            albedoDescriptor = &descriptor;
            break;
          }
        }
        const bool textureBudgetAvailable = kEnableNativeTextureMaterials
          &&
          g_nativeTextureRefs.size() < kMaxNativeTextureRefs
          && g_nativeTextureBytes < kMaxNativeTextureBytes;
        if (albedoDescriptor && textureBudgetAvailable)
          materialKey = XXH64(&albedoDescriptor->resource, sizeof(albedoDescriptor->resource), materialKey);
        for (const auto& descriptor : pipelineInfo->boundDescriptors) {
          if (descriptor.kind == D3D12NativeDescriptorKind::Sampler
           && descriptor.viewDescriptionSize >= sizeof(D3D12_SAMPLER_DESC)) {
            materialKey = XXH64(
              descriptor.viewDescription.data(),
              sizeof(D3D12_SAMPLER_DESC), materialKey);
            break;
          }
        }
        const bool forceWhite = g_forceWhiteMaterials.load(std::memory_order_relaxed) != 0;
        const bool litWhite = forceWhite
          && isAczModelShader(pipelineInfo->vertexShaderHash);
        materialKey = XXH64(&forceWhite, sizeof(forceWhite), materialKey);
        materialKey = XXH64(&litWhite, sizeof(litWhite), materialKey);
        materialKey = (materialKey | 0x1000ull);
        materialHandle = reinterpret_cast<remixapi_MaterialHandle>(static_cast<uintptr_t>(materialKey));
        if (g_registeredMaterials.emplace(materialKey).second) {
          auto opaque = dxvk::LegacyMaterialData::createDefault();

          if (forceWhite) {
            // The white model path: every surface is opaque and untextured, so the
            // captured geometry is visible regardless of the game's textures and
            // lighting. Object geometry stays lit so it keeps a readable shape.
            opaque.setUseLegacyAlphaState(false);
            opaque.setBlendEnabled(false);
            opaque.setAlphaTestType(AlphaTestType::kAlways);
            opaque.setEmissiveIntensity(1.0f);
            // Object geometry is tinted instead of white so that it stays
            // distinguishable from the white terrain and so that a draw which is
            // missing from the frame can be told apart from one that is merely
            // white against white.
            opaque.setEmissiveColorConstant(litWhite
              ? dxvk::Vector3(1.0f, 0.35f, 0.05f)
              : dxvk::Vector3(1.0f, 1.0f, 1.0f));
            {
              static std::atomic<uint32_t> whiteLogs { 0 };
              const uint32_t index = whiteLogs.fetch_add(1, std::memory_order_relaxed);
              if (index < 4) {
                log("D3D12 white model material: vs=0x%016llx selfLit=%u",
                  static_cast<unsigned long long>(pipelineInfo->vertexShaderHash),
                  litWhite ? 0u : 1u);
              }
            }
          } else {
            if (albedoDescriptor) {
              const auto texture = materialTextureFromDescriptor(*albedoDescriptor);
              if (texture.isValid() && !texture.isImageEmpty())
                opaque.setAlbedoOpacityTexture(texture);
            }
            const auto sampler = materialSamplerFromDescriptors(
              pipelineInfo->boundDescriptors);
            if (sampler.ptr())
              opaque.setSamplerOverride(sampler);
            opaque.setUseLegacyAlphaState(true);
            opaque.setBlendEnabled(pipelineInfo->blendEnabled);
            opaque.setAlphaTestType(pipelineInfo->alphaTestEnabled
              ? AlphaTestType::kGreater : AlphaTestType::kAlways);
            opaque.setBlendType(BlendType::kAlpha);
          }
          if (!g_nativeRemixApi.registerExternalMaterial(materialHandle, dxvk::MaterialData { opaque }))
            log("D3D12 native Remix material registration failed: material=%p", materialHandle);
        }
      }
      geometry.externalMaterial = materialHandle;

      const auto handleValue = ++g_sceneMeshSerial;
      auto handle = reinterpret_cast<remixapi_MeshHandle>(
        static_cast<uintptr_t>((handleValue << 4) | 1));
      {
        static uint32_t drawParamLogs = 0;
        if (drawParamLogs < 10 && pipelineInfo->indexCount >= 1000) {
          ++drawParamLogs;
          log("ACZ draw params: baseVertex=%d startIndex=%u indexCount=%u stride=%u ibFormat=%u vbLoc=0x%llx vbSize=%llu",
            pipelineInfo->baseVertexLocation,
            pipelineInfo->startIndexLocation,
            pipelineInfo->indexCount,
            pipelineInfo->vertexBufferStride,
            static_cast<unsigned>(pipelineInfo->indexBufferFormat),
            static_cast<unsigned long long>(pipelineInfo->vertexBufferLocation),
            static_cast<unsigned long long>(pipelineInfo->vertexBufferSize));
        }
      }
      g_meshStorage.push_back({ handle, vertexBuffer, indexBuffer });
      const bool registered = g_nativeRemixApi.registerExternalMesh(
        handle, std::vector<dxvk::RasterGeometry>{ std::move(geometry) });
      if (!registered) {
        log("D3D12 native Remix mesh registration failed: mesh=%p", handle);
        g_meshStorage.pop_back();
        return false;
      }
      g_nativeMeshHandles.emplace(key, handle);

      auto& sceneManager = g_device->getCommon()->getSceneManager();
      const uint32_t instancesBefore = sceneManager.getInstanceManager().getActiveCount();
      if (worldSpaceGeometry)
        g_nativeWorldSpaceMeshes.emplace(key);
      const bool instanceAccepted = commitNativeMeshInstance(
        handle, pipelineInfo, worldSpaceGeometry);
      if (worldSpaceGeometry && instanceAccepted)
        g_nativeWorldSpaceSubmits.fetch_add(1, std::memory_order_relaxed);
      const uint32_t instancesAfter = sceneManager.getInstanceManager().getActiveCount();
      log("D3D12 native Remix submit state: frame=%u cameraValid=%u instances=%u surfaces=%u",
        g_device->getCurrentFrameId(),
        sceneManager.getCameraManager().isCameraValid(dxvk::CameraType::Main) ? 1u : 0u,
        instancesAfter,
        sceneManager.getAccelManager().getSurfaceCount());
      log("D3D12 native Remix scene submit: mesh=%p material=%p registered=%u instanceAccepted=%u vertices=%u indices=%u stride=%u instances=%u",
        handle, materialHandle, registered ? 1u : 0u, instanceAccepted ? 1u : 0u,
        vertexCount, indexCount, vertexStride, instancesAfter - instancesBefore);
      if (pipelineInfo) {
        const auto& m = pipelineInfo->objectToWorld;
        log("D3D12 native Remix draw provenance: draw=%llu frame=%u pso=0x%llx vs=0x%016llx ps=0x%016llx vb=0x%llx/%u/%u ib=0x%llx/%u/%u args=(%u,%u,%u,%d,%u) transformSource=%u hasTransform=%u t=(%.6g,%.6g,%.6g) descriptorSnapshots=%zu",
          static_cast<unsigned long long>(pipelineInfo->drawId),
          g_device->getCurrentFrameId(),
          static_cast<unsigned long long>(pipelineInfo->pipelineState),
          static_cast<unsigned long long>(pipelineInfo->vertexShaderHash),
          static_cast<unsigned long long>(pipelineInfo->pixelShaderHash),
          static_cast<unsigned long long>(pipelineInfo->vertexBufferLocation),
          pipelineInfo->vertexBufferSize, pipelineInfo->vertexBufferStride,
          static_cast<unsigned long long>(pipelineInfo->indexBufferLocation),
          pipelineInfo->indexBufferSize, static_cast<unsigned>(pipelineInfo->indexBufferFormat),
          pipelineInfo->indexCount, pipelineInfo->instanceCount,
          pipelineInfo->startIndexLocation, pipelineInfo->baseVertexLocation,
          pipelineInfo->startInstanceLocation, pipelineInfo->objectTransformSource,
          pipelineInfo->hasObjectToWorld ? 1u : 0u,
          static_cast<double>(m[12]), static_cast<double>(m[13]), static_cast<double>(m[14]),
          pipelineInfo->boundDescriptors.size());

        size_t loggedDescriptors = 0;
        for (const auto& descriptor : pipelineInfo->boundDescriptors) {
          if (descriptor.kind != D3D12NativeDescriptorKind::Srv
           || descriptor.resourceDescription.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER
           || loggedDescriptors >= 4)
            continue;
          log("D3D12 native Remix draw SRV: draw=%llu root=%u offset=%u resource=0x%llx format=%u size=%llux%u mips=%u viewBytes=%u capturedSubresources=%u capturedBytes=%llu",
            static_cast<unsigned long long>(pipelineInfo->drawId),
            descriptor.rootParameterIndex, descriptor.tableOffset,
            static_cast<unsigned long long>(descriptor.resource),
            static_cast<unsigned>(descriptor.format),
            static_cast<unsigned long long>(descriptor.resourceDescription.Width),
            descriptor.resourceDescription.Height,
            descriptor.resourceDescription.MipLevels,
            descriptor.viewDescriptionSize,
            descriptor.capturedSubresources,
            static_cast<unsigned long long>(descriptor.capturedBytes));
          ++loggedDescriptors;
        }
      }
      return instanceAccepted;
    }

    dxvk::DxvkDeviceFeatures makeDeviceFeatures(const dxvk::Rc<dxvk::DxvkAdapter>& adapter) {
      const auto& supported = adapter->features();
      auto enabled = dxvk::DxvkDeviceFeatures {};

      enabled.core.features.geometryShader = VK_TRUE;
      enabled.core.features.robustBufferAccess = VK_TRUE;
      enabled.extRobustness2.robustBufferAccess2 = supported.extRobustness2.robustBufferAccess2;
      enabled.extMemoryPriority.memoryPriority = supported.extMemoryPriority.memoryPriority;
      enabled.extShaderDemoteToHelperInvocation.shaderDemoteToHelperInvocation = supported.extShaderDemoteToHelperInvocation.shaderDemoteToHelperInvocation;
      enabled.extVertexAttributeDivisor.vertexAttributeInstanceRateDivisor = supported.extVertexAttributeDivisor.vertexAttributeInstanceRateDivisor;
      enabled.extVertexAttributeDivisor.vertexAttributeInstanceRateZeroDivisor = supported.extVertexAttributeDivisor.vertexAttributeInstanceRateZeroDivisor;
      enabled.extRobustness2.nullDescriptor = supported.extRobustness2.nullDescriptor;

      enabled.core.features.vertexPipelineStoresAndAtomics = supported.core.features.vertexPipelineStoresAndAtomics;
      enabled.core.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
      enabled.core.features.shaderStorageImageReadWithoutFormat = VK_TRUE;
      enabled.core.features.imageCubeArray = VK_TRUE;
      enabled.core.features.depthClamp = VK_TRUE;
      enabled.core.features.depthBiasClamp = VK_TRUE;
      enabled.core.features.fillModeNonSolid = VK_TRUE;
      enabled.core.features.pipelineStatisticsQuery = supported.core.features.pipelineStatisticsQuery;
      enabled.core.features.sampleRateShading = VK_TRUE;
      enabled.core.features.samplerAnisotropy = supported.core.features.samplerAnisotropy;
      enabled.core.features.shaderClipDistance = VK_TRUE;
      enabled.core.features.shaderCullDistance = VK_TRUE;
      enabled.core.features.textureCompressionBC = VK_TRUE;
      enabled.extDepthClipEnable.depthClipEnable = supported.extDepthClipEnable.depthClipEnable;
      enabled.vulkan12Features.hostQueryReset = supported.vulkan12Features.hostQueryReset;
      enabled.core.features.occlusionQueryPrecise = VK_TRUE;
      enabled.core.features.multiViewport = VK_TRUE;
      enabled.core.features.independentBlend = VK_TRUE;
      enabled.core.features.fullDrawIndexUint32 = VK_TRUE;
      enabled.core.features.depthBounds = supported.core.features.depthBounds;

      if (supported.extCustomBorderColor.customBorderColorWithoutFormat) {
        enabled.extCustomBorderColor.customBorderColors = VK_TRUE;
        enabled.extCustomBorderColor.customBorderColorWithoutFormat = VK_TRUE;
      }

      enabled.extTransformFeedback.transformFeedback = VK_TRUE;
      enabled.vulkan12Features.shaderOutputViewportIndex = VK_TRUE;
      enabled.vulkan12Features.shaderOutputLayer = VK_TRUE;
      enabled.vulkan12Features.drawIndirectCount = VK_TRUE;
      enabled.vulkan12Features.samplerMirrorClampToEdge = VK_TRUE;
      return enabled;
    }
  }

  void initializeD3D12NativeBackend() {
    std::lock_guard lock(g_backendMutex);
    if (g_initialized || g_failed)
      return;

    try {
      dxvk::Rc<dxvk::DxvkInstance> instance = new dxvk::DxvkInstance();
      auto adapter = instance->enumAdapters(0);
      if (adapter == nullptr) {
        log("D3D12 native DXVK backend: no Vulkan adapter was found");
        g_failed = true;
        return;
      }

      // The adapter exposes the already queried Vulkan feature chain. Passing
      // that chain lets the existing DXVK device builder enable the RTX path
      // without creating a D3D9 device or borrowing the game's D3D12 device.
      g_device = adapter->createDevice(instance, makeDeviceFeatures(adapter));
      if (g_device == nullptr) {
        log("D3D12 native DXVK backend: device creation returned null");
        g_failed = true;
        return;
      }

      g_rtxContext = g_device->createRtxContext();
      g_rtxContext->beginRecording(g_device->createCommandList());
      g_device->getCommon()->getSceneManager().initialize(g_rtxContext);
      g_nativeRemixApi.initialize(g_device, g_rtxContext);
      g_sceneInitialized = true;
      // No game lights are submitted, so the fallback light provides all
    // illumination for the captured scene.
    dxvk::LightManager::fallbackLightMode.setDeferred(
      dxvk::LightManager::FallbackLightMode::Always);
    dxvk::LightManager::fallbackLightRadiance.setDeferred(
      dxvk::Vector3(6.0f, 6.0f, 6.0f));
    dxvk::LightManager::fallbackLightDirection.setDeferred(
      dxvk::Vector3(-0.3f, -0.8f, 0.5f));

    g_initialized = true;
      const auto& properties = adapter->deviceProperties();
      log("D3D12 native DXVK backend initialized: adapter=%s vendor=0x%04x device=0x%04x vkDevice=%p",
        properties.deviceName,
        properties.vendorID,
        properties.deviceID,
        g_device->handle());
    } catch (const dxvk::DxvkError& error) {
      log("D3D12 native DXVK backend initialization failed: %s", error.message().c_str());
      g_failed = true;
    } catch (...) {
      log("D3D12 native DXVK backend initialization failed: unknown exception");
      g_failed = true;
    }
  }

  void attachD3D12NativeDevice(ID3D12Device* device) {
    std::lock_guard lock(g_backendMutex);
    if (!device || g_d3d12Device == device)
      return;
    if (g_d3d12Device)
      g_d3d12Device->Release();
    g_d3d12Device = device;
    g_d3d12Device->AddRef();
  }

  void attachD3D12NativeCommandQueue(ID3D12CommandQueue* commandQueue) {
    std::lock_guard lock(g_backendMutex);
    if (!commandQueue || g_d3d12Queue == commandQueue)
      return;
    if (g_d3d12Queue)
      g_d3d12Queue->Release();
    g_d3d12Queue = commandQueue;
    g_d3d12Queue->AddRef();
  }

  void submitD3D12Camera(const float* viewProjection, const float* worldToView) {
    if (!viewProjection)
      return;

    std::lock_guard lock(g_backendMutex);
    if (!g_initialized || !g_sceneInitialized || g_device == nullptr)
      return;

    uint64_t matrixHash = XXH3_64bits(viewProjection, sizeof(float) * 16);
    if (worldToView)
      matrixHash = XXH64(worldToView, sizeof(float) * 16, matrixHash);
    // Remix reports the camera as valid only on the frame it was last updated,
    // and skips ray tracing on frames where it is not valid, so a submission is
    // required once per frame even when the matrices did not change.
    const uint32_t frame = g_device->getCurrentFrameId();
    if (frame == g_lastCameraFrame)
      return;

    dxvk::Matrix4 view;
    dxvk::Matrix4 projection;
    float p00 = 0.0f;
    float p11 = 0.0f;
    float p22 = 0.0f;
    if (!decomposeRowVectorViewProjection(viewProjection, view, projection, p00, p11, p22))
      return;

    // A dedicated world-to-view matrix wins over the one decomposed from the
    // view-projection, because engines often keep the projection fixed while
    // the view changes every frame.
    if (worldToView) {
      dxvk::Matrix4 explicitView;
      if (buildWorldToView(worldToView, explicitView))
        view = explicitView;
    }

    g_nativeRemixApi.processExternalCamera(
      dxvk::CameraType::Main, view, projection);
    g_cameraWorldToView = view;
    g_cameraViewToProjection = projection;
    g_cameraEye = dxvk::inverse(view)[3].xyz();
    g_hasCameraMatrices = true;
    g_lastCameraHash = matrixHash;
    g_lastCameraFrame = frame;
    auto& cameraManager = g_device->getCommon()->getSceneManager().getCameraManager();
    log("D3D12 native Remix camera updated: frame=%u p=(%.6g,%.6g,%.6g) position=(%.6g,%.6g,%.6g) valid=%u",
      frame,
      static_cast<double>(p00), static_cast<double>(p11), static_cast<double>(p22),
      static_cast<double>(view[3][0]),
      static_cast<double>(view[3][1]),
      static_cast<double>(view[3][2]),
      cameraManager.isCameraValid(dxvk::CameraType::Main) ? 1u : 0u);
  }

  void submitD3D12CameraMatrices(
      const float* worldToView,
      const float* viewToProjection) {
    if (!worldToView || !viewToProjection)
      return;

    std::lock_guard lock(g_backendMutex);
    if (!g_initialized || !g_sceneInitialized || g_device == nullptr)
      return;

    log("D3D12 native Remix camera input: addr=%p t=(%.1f,%.1f,%.1f) row0=(%.3f,%.3f,%.3f)",
      static_cast<const void*>(worldToView),
      static_cast<double>(worldToView[12]),
      static_cast<double>(worldToView[13]),
      static_cast<double>(worldToView[14]),
      static_cast<double>(worldToView[0]),
      static_cast<double>(worldToView[1]),
      static_cast<double>(worldToView[2]));

    dxvk::Matrix4 view;
    if (!buildWorldToView(worldToView, view))
      return;

    log("D3D12 native Remix camera after-build: t=(%.1f,%.1f,%.1f) in=(%.1f,%.1f,%.1f)",
      static_cast<double>(view[3][0]),
      static_cast<double>(view[3][1]),
      static_cast<double>(view[3][2]),
      static_cast<double>(worldToView[12]),
      static_cast<double>(worldToView[13]),
      static_cast<double>(worldToView[14]));

    dxvk::Matrix4 projection;
    for (uint32_t row = 0; row < 4; ++row) {
      for (uint32_t column = 0; column < 4; ++column)
        projection[row][column] = viewToProjection[row * 4 + column];
    }

    uint64_t matrixHash = XXH3_64bits(worldToView, sizeof(float) * 16);
    matrixHash = XXH64(viewToProjection, sizeof(float) * 16, matrixHash);
    // See the note in submitD3D12Camera: one submission per frame is required
    // for the camera to stay valid, even when the matrices are unchanged.
    const uint32_t frame = g_device->getCurrentFrameId();
    if (frame == g_lastCameraFrame)
      return;

    g_nativeRemixApi.processExternalCamera(
      dxvk::CameraType::Main, view, projection);
    g_cameraWorldToView = view;
    g_cameraViewToProjection = projection;
    g_cameraEye = dxvk::inverse(view)[3].xyz();
    g_hasCameraMatrices = true;
    g_lastCameraHash = matrixHash;
    g_lastCameraFrame = frame;

    log("D3D12 native Remix camera direct: frame=%u position=(%.6g,%.6g,%.6g)",
      frame,
      static_cast<double>(view[3][0]),
      static_cast<double>(view[3][1]),
      static_cast<double>(view[3][2]));
  }

  void setD3D12OutputEnabled(bool enabled) {
    g_remixOutputEnabled.store(enabled, std::memory_order_relaxed);
    log("D3D12 native Remix output toggled: enabled=%u", enabled ? 1u : 0u);
  }

  void setD3D12WorldSpacePolicy(uint32_t mode, float minExtent) {
    const uint32_t clamped = mode > static_cast<uint32_t>(WorldSpacePolicy::AcceptAll)
      ? static_cast<uint32_t>(WorldSpacePolicy::Reject)
      : mode;
    g_worldSpacePolicy.store(clamped, std::memory_order_relaxed);
    if (minExtent > 0.0f)
      g_worldSpaceMinExtent.store(minExtent, std::memory_order_relaxed);
    log("D3D12 world-space policy: mode=%u minExtent=%.3f",
      clamped, static_cast<double>(g_worldSpaceMinExtent.load(std::memory_order_relaxed)));
  }

  void beginD3D12NativeFrame(
      void* swapChain,
      UINT backBufferIndex,
      ID3D12Resource* backBuffer,
      const D3D12_RESOURCE_DESC* backBufferDescription) {
    std::lock_guard lock(g_backendMutex);
    if (!g_initialized || g_device == nullptr)
      return;

    const bool outputKeyDown = (GetAsyncKeyState(VK_F10) & 0x8000) != 0;
    if (outputKeyDown && !g_remixOutputKeyDown) {
      const bool enabled = !g_remixOutputEnabled.load(std::memory_order_relaxed);
      g_remixOutputEnabled.store(enabled, std::memory_order_relaxed);
      log("D3D12 native Remix output toggled: enabled=%u", enabled ? 1u : 0u);
    }
    g_remixOutputKeyDown = outputKeyDown;

    const bool whiteKeyDown = (GetAsyncKeyState(VK_F7) & 0x8000) != 0;
    if (whiteKeyDown && !g_whiteMaterialKeyDown) {
      const bool enabled = g_forceWhiteMaterials.load(std::memory_order_relaxed) == 0;
      g_forceWhiteMaterials.store(enabled ? 1u : 0u, std::memory_order_relaxed);
      log("ACZ white model materials: enabled=%u", enabled ? 1u : 0u);
    }
    g_whiteMaterialKeyDown = whiteKeyDown;

    if (!g_remixOutputEnabled.load(std::memory_order_relaxed)) {
      const uint32_t frame = g_device->getCurrentFrameId();
      if ((frame % 120) == 0)
        log("D3D12 native Remix output disabled: native D3D12 passthrough frame=%u", frame);
      return;
    }

    if (backBufferDescription) {
      log("D3D12 native Remix frame begin: swapChain=%p backBufferIndex=%u resource=%p width=%llu height=%u format=%u",
        swapChain,
        backBufferIndex,
        backBuffer,
        static_cast<unsigned long long>(backBufferDescription->Width),
        backBufferDescription->Height,
        static_cast<unsigned>(backBufferDescription->Format));
    } else {
      log("D3D12 native Remix frame begin: swapChain=%p backBufferIndex=%u resource=unavailable output=unbound",
        swapChain, backBufferIndex);
    }

    const bool outputImported = importBackBuffer(
      swapChain, backBufferIndex, backBuffer, backBufferDescription);

    if (g_hasCameraMatrices) {
      g_nativeRemixApi.processExternalCamera(
        dxvk::CameraType::Main, g_cameraWorldToView, g_cameraViewToProjection);
    }

    bool remixSceneReady = false;
    uint32_t remixCameraValid = 0;
    uint32_t remixInstanceCount = 0;
    uint32_t remixSurfaceCount = 0;
    bool externalSyncArmed = false;
    if (g_rtxContext.ptr() && g_sceneInitialized) {
      // External D3D12 draws are queued into the Remix scene manager, but the
      // native D3D12 path has no DXVK injectRTX call to prepare the scene for
      // us. Build the BLAS/TLAS and surface/material buffers before flushing
      // this independent command context.
      auto& sceneManager = g_device->getCommon()->getSceneManager();
      remixCameraValid = sceneManager.getCameraManager().isCameraValid(dxvk::CameraType::Main) ? 1u : 0u;
      remixInstanceCount = sceneManager.getInstanceManager().getActiveCount();
      remixSurfaceCount = sceneManager.getAccelManager().getSurfaceCount();
      const bool preservePreviousScene =
        g_hasPreparedScene && remixCameraValid && remixInstanceCount == 0;

      if (remixCameraValid && (remixInstanceCount || preservePreviousScene)
       && initializeExternalFrameSync()) {
        const uint64_t inputValue = ++g_externalFrameSync.inputValue;
        HRESULT signalResult = g_d3d12Queue->Signal(
          g_externalFrameSync.inputFence, inputValue);
        if (SUCCEEDED(signalResult)) {
          g_rtxContext->getCommandList()->addWaitSemaphore(
            g_externalFrameSync.inputSemaphore, inputValue);
          externalSyncArmed = true;
          log("D3D12 native Remix external sync input: value=%llu",
            static_cast<unsigned long long>(inputValue));
        } else {
          log("D3D12 native Remix external sync input signal failed: result=0x%08lx",
            static_cast<unsigned long>(signalResult));
        }
      }
      if (preservePreviousScene) {
        ++g_preservedSceneFrames;
        remixSceneReady = true;
        log("D3D12 native Remix scene preserve: frame=%u cameraValid=%u instances=%u surfaces=%u grace=%u",
          g_device->getCurrentFrameId(), remixCameraValid, remixInstanceCount,
          remixSurfaceCount, g_preservedSceneFrames);
      } else {
        g_preservedSceneFrames = 0;
        log("D3D12 native Remix prepare begin: frame=%u cameraValid=%u instances=%u surfaces=%u",
          g_device->getCurrentFrameId(),
          remixCameraValid, remixInstanceCount, remixSurfaceCount);
        {
          PerfScope prepareScope { g_nativePrepareTicks, g_nativePrepareCalls };
          g_nativeRemixApi.prepareScene();
        }
        const auto& instances = sceneManager.getInstanceTable();
        remixCameraValid = sceneManager.getCameraManager().isCameraValid(dxvk::CameraType::Main) ? 1u : 0u;
        remixInstanceCount = sceneManager.getInstanceManager().getActiveCount();
        remixSurfaceCount = sceneManager.getAccelManager().getSurfaceCount();
        remixSceneReady = remixCameraValid != 0 && remixInstanceCount != 0 && remixSurfaceCount != 0;
        if (!remixSceneReady && g_hasPreparedScene && remixCameraValid && remixSurfaceCount != 0) {
          // A draw can submit a transient instance immediately before prepare,
          // then retire it while the persistent surface/TLAS remains valid.
          // Keep the previous scene for this frame instead of switching the
          // shared output back to the native D3D12 path.
          remixSceneReady = true;
          ++g_preservedSceneFrames;
          log("D3D12 native Remix scene fallback: frame=%u cameraValid=%u instances=%u surfaces=%u grace=%u",
            g_device->getCurrentFrameId(), remixCameraValid, remixInstanceCount,
            remixSurfaceCount, g_preservedSceneFrames);
        }
        if (remixSceneReady)
          g_hasPreparedScene = true;
        else if (!remixCameraValid)
          g_hasPreparedScene = false;
        log("D3D12 native Remix prepare end: frame=%u cameraValid=%u instances=%u surfaces=%u",
          g_device->getCurrentFrameId(),
          remixCameraValid, remixInstanceCount, remixSurfaceCount);
        log("D3D12 native Remix scene state: meshes=%llu instances=%zu surfaces=%u",
          static_cast<unsigned long long>(g_sceneMeshSerial), instances.size(),
          g_device->getCommon()->getSceneManager().getAccelManager().getSurfaceCount());
      }
    }

    if (outputImported && g_outputImage.ptr() && remixSceneReady && externalSyncArmed) {
      try {
        PerfScope injectScope { g_nativeInjectTicks, g_nativeInjectCalls };
        g_rtxContext->injectRTX(g_device->getCurrentFrameId(), g_outputImage);
        const uint64_t outputValue = ++g_externalFrameSync.outputValue;
        g_rtxContext->getCommandList()->addSignalSemaphore(
          g_externalFrameSync.outputSemaphore, outputValue);
        g_rtxContext->flushCommandList();
        HRESULT waitResult = g_d3d12Queue->Wait(
          g_externalFrameSync.outputFence, outputValue);
        if (FAILED(waitResult)) {
          log("D3D12 native Remix external sync output wait failed: value=%llu result=0x%08lx",
            static_cast<unsigned long long>(outputValue),
            static_cast<unsigned long>(waitResult));
        } else {
          log("D3D12 native Remix external sync output: value=%llu",
            static_cast<unsigned long long>(outputValue));
        }
        log("D3D12 native Remix output handoff: state=bound mode=shared-d3d12-image");
      } catch (const dxvk::DxvkError& error) {
        log("D3D12 native Remix output handoff failed: %s", error.message().c_str());
        g_outputImage = nullptr;
      } catch (...) {
        log("D3D12 native Remix output handoff failed: unknown exception");
        g_outputImage = nullptr;
      }
    } else {
      log("D3D12 native Remix output handoff: state=unbound reason=%s cameraValid=%u instances=%u surfaces=%u",
        outputImported && g_outputImage.ptr()
          ? (remixSceneReady ? "external-sync-unavailable" : "no-valid-remix-scene")
          : "back-buffer-not-shareable",
        remixCameraValid, remixInstanceCount, remixSurfaceCount);
    }
  }

  void recordD3D12NativePresent() {
    std::lock_guard lock(g_backendMutex);
    if (!g_initialized || g_device == nullptr)
      return;

    // Advance the DXVK frame only after the game's Present has returned. This
    // keeps all draws captured before Present in the frame that was prepared
    // above and prevents scene resources from being collected too early.
    g_device->incrementPresentCount();
    const uint32_t frame = g_device->getCurrentFrameId();
    if (frame % 60 == 0 && g_perfFrequency.QuadPart) {
      const auto toMs = [](uint64_t ticks, LONGLONG frequency) {
        return static_cast<double>(ticks) * 1000.0
          / static_cast<double>(frequency);
      };
      log("D3D12 native performance window: frames=60 submitCalls=%llu submitMs=%.3f prepareCalls=%llu prepareMs=%.3f injectCalls=%llu injectMs=%.3f",
        static_cast<unsigned long long>(g_nativeSubmitCalls.exchange(0, std::memory_order_relaxed)),
        toMs(g_nativeSubmitTicks.exchange(0, std::memory_order_relaxed), g_perfFrequency.QuadPart),
        static_cast<unsigned long long>(g_nativePrepareCalls.exchange(0, std::memory_order_relaxed)),
        toMs(g_nativePrepareTicks.exchange(0, std::memory_order_relaxed), g_perfFrequency.QuadPart),
        static_cast<unsigned long long>(g_nativeInjectCalls.exchange(0, std::memory_order_relaxed)),
        toMs(g_nativeInjectTicks.exchange(0, std::memory_order_relaxed), g_perfFrequency.QuadPart));
      log("D3D12 native rejection window: noTransform=%llu identityFiltered=%llu worldSpaceFallbacks=%llu worldSpaceSubmits=%llu worldSpaceInstances=%llu readFailures=%llu conversionFailures=%llu acceptedInstances=%llu",
        static_cast<unsigned long long>(g_nativeNoTransform.exchange(0, std::memory_order_relaxed)),
        static_cast<unsigned long long>(g_nativeIdentityFiltered.exchange(0, std::memory_order_relaxed)),
        static_cast<unsigned long long>(g_nativeWorldSpaceFallbacks.exchange(0, std::memory_order_relaxed)),
        static_cast<unsigned long long>(g_nativeWorldSpaceSubmits.exchange(0, std::memory_order_relaxed)),
        static_cast<unsigned long long>(g_nativeWorldSpaceInstances.exchange(0, std::memory_order_relaxed)),
        static_cast<unsigned long long>(g_nativeReadFailures.exchange(0, std::memory_order_relaxed)),
        static_cast<unsigned long long>(g_nativeConversionFailures.exchange(0, std::memory_order_relaxed)),
        static_cast<unsigned long long>(g_nativeAcceptedInstances.exchange(0, std::memory_order_relaxed)));
      log("D3D12 native camera window: frame=%u position=(%.4f,%.4f,%.4f) enabled=%u policy=%u",
        frame,
        static_cast<double>(g_cameraWorldToView[3][0]),
        static_cast<double>(g_cameraWorldToView[3][1]),
        static_cast<double>(g_cameraWorldToView[3][2]),
        g_remixOutputEnabled.load(std::memory_order_relaxed) ? 1u : 0u,
        g_worldSpacePolicy.load(std::memory_order_relaxed));
      {
        const float nearestSq = std::sqrt(
          g_nativeNearestInstanceDistanceSq.exchange(3.4e38f, std::memory_order_relaxed));
        const uint32_t nearInstances =
          g_nativeNearInstances.exchange(0, std::memory_order_relaxed);
        log("ACZ instance distance window: eye=(%.1f,%.1f,%.1f) nearest=%.1f nearInstances=%u",
          static_cast<double>(g_cameraEye.x),
          static_cast<double>(g_cameraEye.y),
          static_cast<double>(g_cameraEye.z),
          static_cast<double>(nearestSq),
          nearInstances);
      }
      auto& windowSceneManager = g_device->getCommon()->getSceneManager();
      log("D3D12 native scene window: meshes=%llu instances=%u surfaces=%u",
        static_cast<unsigned long long>(g_sceneMeshSerial),
        windowSceneManager.getInstanceManager().getActiveCount(),
        windowSceneManager.getAccelManager().getSurfaceCount());
    }
  }

  void submitD3D12NativeGeometry(
      const D3D12_VERTEX_BUFFER_VIEW& vertexBuffer,
      const D3D12_INDEX_BUFFER_VIEW& indexBuffer,
      D3D12_PRIMITIVE_TOPOLOGY topology,
      UINT indexCount,
      UINT instanceCount,
      UINT startIndexLocation,
      INT baseVertexLocation,
      UINT startInstanceLocation,
      const D3D12NativePipelineInfo* pipelineInfo) {
    std::lock_guard lock(g_backendMutex);
    PerfScope submitScope { g_nativeSubmitTicks, g_nativeSubmitCalls };
    noteAircraftDraw(pipelineInfo ? pipelineInfo->vertexShaderHash : 0ull,
      kAircraftDraw);
    if (!g_initialized || !g_sceneInitialized || !vertexBuffer.BufferLocation
     || !vertexBuffer.StrideInBytes
     || instanceCount == 0 || startInstanceLocation != 0)
      return;

    // Non-indexed draws carry their vertex count instead of an index buffer.
    const bool implicitIndices = pipelineInfo && pipelineInfo->implicitIndices;
    const DXGI_FORMAT effectiveIndexFormat = implicitIndices
      ? DXGI_FORMAT_R32_UINT
      : indexBuffer.Format;
    UINT effectiveIndexSize = indexBuffer.SizeInBytes;
    if (implicitIndices) {
      indexCount = pipelineInfo->vertexCount;
      effectiveIndexSize = indexCount * sizeof(uint32_t);
    } else if (!indexBuffer.BufferLocation || !indexCount) {
      return;
    }

    const uint32_t sourceIndexStride = indexStride(effectiveIndexFormat);
    const VkPrimitiveTopology sourceTopology = toVkTopology(topology);
    if (!sourceIndexStride || (sourceTopology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
       && sourceTopology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP)
     || indexCount < 3 || indexCount > 1'000'000
     || vertexBuffer.SizeInBytes > kMaxNativeGeometryBytes) {
      // Unsupported topologies must not vanish silently: the patch list values
      // used by tessellated draws are the ones that reach this branch.
      static std::mutex topologyMutex;
      static std::unordered_set<uint32_t> loggedTopologies;
      noteAircraftDraw(pipelineInfo ? pipelineInfo->vertexShaderHash : 0ull,
        kAircraftTopology);
      if (sourceIndexStride) {
        std::lock_guard lock(topologyMutex);
        if (loggedTopologies.insert(static_cast<uint32_t>(topology)).second) {
          log("D3D12 geometry bridge dropped draw: reason=topology topology=%u indices=%u stride=%u vs=0x%016llx",
            static_cast<unsigned>(topology), indexCount,
            vertexBuffer.StrideInBytes,
            static_cast<unsigned long long>(
              pipelineInfo ? pipelineInfo->vertexShaderHash : 0ull));
        }
      }
      return;
    }

    uint64_t requestKey = vertexBuffer.BufferLocation;
    requestKey = XXH64(&indexBuffer.BufferLocation, sizeof(indexBuffer.BufferLocation), requestKey);
    requestKey = XXH64(&vertexBuffer.SizeInBytes, sizeof(vertexBuffer.SizeInBytes), requestKey);
    requestKey = XXH64(&vertexBuffer.StrideInBytes, sizeof(vertexBuffer.StrideInBytes), requestKey);
    requestKey = XXH64(&effectiveIndexSize, sizeof(effectiveIndexSize), requestKey);
    requestKey = XXH64(&effectiveIndexFormat, sizeof(effectiveIndexFormat), requestKey);
    requestKey = XXH64(&indexCount, sizeof(indexCount), requestKey);
    requestKey = XXH64(&instanceCount, sizeof(instanceCount), requestKey);
    requestKey = XXH64(&startIndexLocation, sizeof(startIndexLocation), requestKey);
    requestKey = XXH64(&baseVertexLocation, sizeof(baseVertexLocation), requestKey);
    requestKey = XXH64(&startInstanceLocation, sizeof(startInstanceLocation), requestKey);
    requestKey = XXH64(&topology, sizeof(topology), requestKey);
    if (pipelineInfo) {
      requestKey = XXH64(&pipelineInfo->vertexShaderHash,
        sizeof(pipelineInfo->vertexShaderHash), requestKey);
      requestKey = XXH64(&pipelineInfo->pixelShaderHash,
        sizeof(pipelineInfo->pixelShaderHash), requestKey);
    }
    auto cachedRequest = g_nativeRequestHandles.find(requestKey);
    if (cachedRequest != g_nativeRequestHandles.end()) {
      commitNativeMeshInstance(cachedRequest->second, pipelineInfo,
        g_nativeWorldSpaceRequests.find(requestKey)
          != g_nativeWorldSpaceRequests.end());
      return;
    }

    std::vector<uint8_t> sourceIndices;
    const size_t indexBytes = static_cast<size_t>(indexCount) * sourceIndexStride;
    if (implicitIndices) {
      sourceIndices.resize(indexBytes);
      for (UINT i = 0; i < indexCount; ++i) {
        const uint32_t value = i;
        std::memcpy(sourceIndices.data() + static_cast<size_t>(i) * sourceIndexStride,
          &value, sizeof(value));
      }
    } else if (!readGpuBuffer(
        indexBuffer.BufferLocation + static_cast<uint64_t>(startIndexLocation) * sourceIndexStride,
        indexBytes, sourceIndices)) {
      g_nativeReadFailures.fetch_add(1, std::memory_order_relaxed);
      noteAircraftDraw(pipelineInfo ? pipelineInfo->vertexShaderHash : 0ull,
        kAircraftUnreadable);
      return;
    }

    std::vector<uint32_t> normalizedIndices;
    normalizedIndices.reserve(sourceTopology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP
      ? static_cast<size_t>(indexCount - 2) * 3
      : indexCount);
    uint32_t maxIndex = 0;
    const uint32_t restartIndex = sourceIndexStride == 2 ? 0xFFFFu : 0xFFFFFFFFu;
    auto appendIndex = [&](uint32_t raw) {
      const int64_t normalized = static_cast<int64_t>(raw) + baseVertexLocation;
      if (normalized < 0 || normalized > std::numeric_limits<uint32_t>::max())
        return false;
      normalizedIndices.push_back(static_cast<uint32_t>(normalized));
      maxIndex = std::max(maxIndex, static_cast<uint32_t>(normalized));
      return true;
    };

    if (sourceTopology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST) {
      for (UINT i = 0; i < indexCount; ++i) {
        const uint32_t raw = readIndex(sourceIndices, sourceIndexStride, i);
        if (raw == restartIndex || !appendIndex(raw)) {
          logBridgeStageOnce("list index");
          return;
        }
      }
    } else {
      std::vector<uint32_t> strip;
      strip.reserve(indexCount);
      auto flushStrip = [&]() {
        for (size_t i = 2; i < strip.size(); ++i) {
          const uint32_t a = strip[i - 2];
          const uint32_t b = strip[i - 1];
          const uint32_t c = strip[i];
          if (a == b || b == c || a == c)
            continue;
          if ((i & 1) == 0) {
            normalizedIndices.push_back(a);
            normalizedIndices.push_back(b);
          } else {
            normalizedIndices.push_back(b);
            normalizedIndices.push_back(a);
          }
          normalizedIndices.push_back(c);
          maxIndex = std::max(maxIndex, std::max(a, std::max(b, c)));
        }
        strip.clear();
      };

      for (UINT i = 0; i < indexCount; ++i) {
        const uint32_t raw = readIndex(sourceIndices, sourceIndexStride, i);
        if (raw == restartIndex) {
          flushStrip();
          continue;
        }
        const int64_t normalized = static_cast<int64_t>(raw) + baseVertexLocation;
        if (normalized < 0 || normalized > std::numeric_limits<uint32_t>::max())
          return;
        strip.push_back(static_cast<uint32_t>(normalized));
      }
      flushStrip();
    }

    const uint32_t normalizedIndexCount = static_cast<uint32_t>(normalizedIndices.size());
    if (normalizedIndexCount < 3 || (normalizedIndexCount % 3) != 0) {
      logBridgeStageOnce("normalized count");
      return;
    }

    const uint64_t vertexCount64 = static_cast<uint64_t>(maxIndex) + 1;
    if (vertexCount64 == 0 || vertexCount64 > vertexBuffer.SizeInBytes / vertexBuffer.StrideInBytes) {
      logBridgeStageOnce("vertex count");
      return;
    }
    const size_t vertexBytes = static_cast<size_t>(vertexCount64) * vertexBuffer.StrideInBytes;
    if (vertexBytes > kMaxNativeGeometryBytes)
      return;

    std::vector<uint8_t> vertexData;
    if (!readGpuBuffer(vertexBuffer.BufferLocation, vertexBytes, vertexData)) {
      g_nativeReadFailures.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    std::vector<uint8_t> indexData(normalizedIndices.size() * sizeof(uint32_t));
    std::memcpy(indexData.data(), normalizedIndices.data(), indexData.size());
    const uint64_t key = geometryKey(vertexData, indexData, vertexBuffer.StrideInBytes, normalizedIndexCount, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    uint64_t materialLayoutKey = key;
    if (pipelineInfo) {
      materialLayoutKey = XXH64(&pipelineInfo->vertexShaderHash,
        sizeof(pipelineInfo->vertexShaderHash), materialLayoutKey);
      materialLayoutKey = XXH64(&pipelineInfo->pixelShaderHash,
        sizeof(pipelineInfo->pixelShaderHash), materialLayoutKey);
    }
    const bool meshSubmitted = submitNativeMesh(vertexData, indexData, vertexBuffer.StrideInBytes,
      static_cast<uint32_t>(vertexCount64), normalizedIndexCount, VK_INDEX_TYPE_UINT32,
      VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, materialLayoutKey, pipelineInfo);
    if (meshSubmitted)
    {
      auto mesh = g_nativeMeshHandles.find(materialLayoutKey);
      if (mesh != g_nativeMeshHandles.end()) {
        g_nativeRequestHandles.emplace(requestKey, mesh->second);
        if (g_nativeWorldSpaceMeshes.find(materialLayoutKey)
            != g_nativeWorldSpaceMeshes.end())
          g_nativeWorldSpaceRequests.emplace(requestKey);
      }
    }
  }

  bool readD3D12InstanceBounds(
      float& centerX, float& centerY, float& centerZ, float& radius) {
    std::lock_guard lock(g_boundsMutex);
    if (g_boundsCount < 2)
      return false;

    centerX = (g_boundsMin[0] + g_boundsMax[0]) * 0.5f;
    centerY = (g_boundsMin[1] + g_boundsMax[1]) * 0.5f;
    centerZ = (g_boundsMin[2] + g_boundsMax[2]) * 0.5f;
    const float dx = g_boundsMax[0] - g_boundsMin[0];
    const float dy = g_boundsMax[1] - g_boundsMin[1];
    const float dz = g_boundsMax[2] - g_boundsMin[2];
    radius = std::sqrt(dx * dx + dy * dy + dz * dz) * 0.5f;
    return radius > 0.0f;
  }

  bool readD3D12CameraViewToWorld(std::array<float, 16>& out) {
    std::lock_guard lock(g_backendMutex);
    if (!g_hasCameraMatrices)
      return false;
    const dxvk::Matrix4 viewToWorld = dxvk::inverse(g_cameraWorldToView);
    for (uint32_t row = 0; row < 4; ++row) {
      for (uint32_t column = 0; column < 4; ++column)
        out[row * 4 + column] = viewToWorld[row][column];
    }
    return true;
  }

}
