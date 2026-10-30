#include "d3d12_device_extension_trace.h"

#include "d3d12_command_list_trace.h"
#include "d3d12_device_proxy.h"

#include <windows.h>

#include <cstring>
#include <mutex>
#include <unordered_map>

namespace remix {

  namespace {

    // IUnknown + ID3D12Device through ID3D12Device3 + ID3D12Device4 methods.
    constexpr size_t kDevice4MethodCount = 57;
    constexpr size_t kQueryInterfaceIndex = 0;
    constexpr size_t kCreateCommandListIndex = 12;
    constexpr size_t kCreateCommandList1Index = 51;

    struct TraceEntry {
      void** vtable;
      void* queryInterface;
      void* createCommandList;
      void* createCommandList1;
    };

    std::mutex g_traceMutex;
    std::unordered_map<ID3D12Device4*, TraceEntry> g_traceEntries;

    TraceEntry* findEntry(ID3D12Device4* device) {
      std::lock_guard lock(g_traceMutex);
      auto entry = g_traceEntries.find(device);
      return entry == g_traceEntries.end() ? nullptr : &entry->second;
    }

    HRESULT STDMETHODCALLTYPE traceQueryInterface(
        ID3D12Device4* device,
        REFIID riid,
        void** object) {
      auto* entry = findEntry(device);
      auto function = reinterpret_cast<HRESULT (STDMETHODCALLTYPE*)(
        ID3D12Device4*, REFIID, void**)>(entry->queryInterface);
      HRESULT result = function(device, riid, object);
      bool supported = isDevice4OrLaterIid(riid);
      bool installed = false;
      if (SUCCEEDED(result) && supported && object && *object)
        installed = installDevice4Trace(static_cast<ID3D12Device4*>(*object));
      log("ID3D12Device4::QueryInterface(riid=%s, result=0x%08lx, object=%p, traceSupported=%u, traceInstalled=%u)",
        formatIid(riid), static_cast<unsigned long>(result),
        object ? *object : nullptr, supported ? 1u : 0u, installed ? 1u : 0u);
      return result;
    }

    HRESULT STDMETHODCALLTYPE traceCreateCommandList1(
        ID3D12Device4* device,
        UINT nodeMask,
        D3D12_COMMAND_LIST_TYPE type,
        D3D12_COMMAND_LIST_FLAGS flags,
        REFIID riid,
        void** commandList) {
      auto* entry = findEntry(device);
      auto function = reinterpret_cast<HRESULT (STDMETHODCALLTYPE*)(
        ID3D12Device4*, UINT, D3D12_COMMAND_LIST_TYPE,
        D3D12_COMMAND_LIST_FLAGS, REFIID, void**)>(entry->createCommandList1);
      HRESULT result = function(device, nodeMask, type, flags, riid, commandList);
      bool traceSupported = isGraphicsCommandListIid(riid);
      bool traceInstalled = false;
      if (SUCCEEDED(result) && traceSupported && commandList && *commandList)
        traceInstalled = installCommandListTrace(
          static_cast<ID3D12GraphicsCommandList*>(*commandList));
      log("ID3D12Device4::CreateCommandList1(node=%u, type=%u, flags=0x%x, riid=%s, result=0x%08lx, object=%p, traceSupported=%u, traceInstalled=%u)",
        nodeMask, static_cast<unsigned>(type), flags, formatIid(riid),
        static_cast<unsigned long>(result), commandList ? *commandList : nullptr,
        traceSupported ? 1u : 0u, traceInstalled ? 1u : 0u);
      return result;
    }

    HRESULT STDMETHODCALLTYPE traceCreateCommandList(
        ID3D12Device4* device,
        UINT nodeMask,
        D3D12_COMMAND_LIST_TYPE type,
        ID3D12CommandAllocator* commandAllocator,
        ID3D12PipelineState* initialState,
        REFIID riid,
        void** commandList) {
      auto* entry = findEntry(device);
      auto function = reinterpret_cast<HRESULT (STDMETHODCALLTYPE*)(
        ID3D12Device4*, UINT, D3D12_COMMAND_LIST_TYPE,
        ID3D12CommandAllocator*, ID3D12PipelineState*, REFIID, void**)>(
          entry->createCommandList);
      HRESULT result = function(device, nodeMask, type, commandAllocator,
        initialState, riid, commandList);
      bool traceSupported = isGraphicsCommandListIid(riid);
      bool traceInstalled = false;
      if (SUCCEEDED(result) && traceSupported && commandList && *commandList)
        traceInstalled = installCommandListTrace(
          static_cast<ID3D12GraphicsCommandList*>(*commandList));
      log("ID3D12Device4::CreateCommandList(node=%u, type=%u, riid=%s, result=0x%08lx, object=%p, traceSupported=%u, traceInstalled=%u)",
        nodeMask, static_cast<unsigned>(type), formatIid(riid),
        static_cast<unsigned long>(result), commandList ? *commandList : nullptr,
        traceSupported ? 1u : 0u, traceInstalled ? 1u : 0u);
      return result;
    }

  }

  bool isDevice4OrLaterIid(REFIID iid) {
    return iid == __uuidof(ID3D12Device4)
      || iid == __uuidof(ID3D12Device5)
      || iid == __uuidof(ID3D12Device6)
      || iid == __uuidof(ID3D12Device7)
      || iid == __uuidof(ID3D12Device8)
      || iid == __uuidof(ID3D12Device9)
      || iid == __uuidof(ID3D12Device10)
      || iid == __uuidof(ID3D12Device11)
      || iid == __uuidof(ID3D12Device12)
      || iid == __uuidof(ID3D12Device13)
      || iid == __uuidof(ID3D12Device14);
  }

  bool installDevice4Trace(ID3D12Device4* device) {
    if (!device) {
      log("Device4 trace installation failed: null device");
      return false;
    }

    {
      std::lock_guard lock(g_traceMutex);
      if (g_traceEntries.find(device) != g_traceEntries.end())
        return true;
    }

    void** originalVtable = *reinterpret_cast<void***>(device);
    auto* clone = static_cast<void**>(VirtualAlloc(
      nullptr,
      kDevice4MethodCount * sizeof(void*),
      MEM_COMMIT | MEM_RESERVE,
      PAGE_READWRITE));
    if (!clone) {
      log("Device4 trace installation failed: VirtualAlloc error=%lu", GetLastError());
      return false;
    }

    std::memcpy(clone, originalVtable, kDevice4MethodCount * sizeof(void*));
    TraceEntry entry {
      clone,
      clone[kQueryInterfaceIndex],
      clone[kCreateCommandListIndex],
      clone[kCreateCommandList1Index],
    };
    clone[kQueryInterfaceIndex] = reinterpret_cast<void*>(&traceQueryInterface);
    clone[kCreateCommandListIndex] = reinterpret_cast<void*>(&traceCreateCommandList);
    clone[kCreateCommandList1Index] = reinterpret_cast<void*>(&traceCreateCommandList1);

    {
      std::lock_guard lock(g_traceMutex);
      g_traceEntries.emplace(device, entry);
    }

    DWORD protection = 0;
    if (!VirtualProtect(reinterpret_cast<void*>(device), sizeof(void*), PAGE_READWRITE, &protection)) {
      log("Device4 trace installation failed: VirtualProtect error=%lu", GetLastError());
      VirtualFree(clone, 0, MEM_RELEASE);
      return false;
    }

    *reinterpret_cast<void***>(device) = clone;
    VirtualProtect(reinterpret_cast<void*>(device), sizeof(void*), protection, &protection);
    log("Installed ID3D12Device4 trace at %p", device);
    return true;
  }

}
