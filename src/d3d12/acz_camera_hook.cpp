#include "acz_camera_hook.h"

#include "d3d12_device_proxy.h"

#include <windows.h>

#include <cstdint>
#include <cstring>
#include <initializer_list>

namespace remix {

  namespace {

    // The probe hooks the renderer entry point that runs every frame, so it also
    // fires while flying. Only pointers known to be valid are recorded; deeper
    // object walks are done from a debugger until the layout is established.
    constexpr uint8_t kPrologue[] = {
      0x48, 0x89, 0x5C, 0x24, 0x08, // mov [rsp+8], rbx
      0x57,                         // push rdi
      0x48, 0x83, 0xEC, 0x20,       // sub rsp, 20h
      0x48, 0x63, 0x41, 0x0C,       // movsxd rax, [rcx+0Ch]
    };

    constexpr uint8_t kSignature[] = {
      0x48, 0x89, 0x5C, 0x24, 0x08, 0x57, 0x48, 0x83, 0xEC,
      0x20, 0x48, 0x63, 0x41, 0x0C, 0x48, 0x8B, 0xDA, 0x48,
      0x8B, 0xF9,
    };

    // Stub state lives in the detour page so every displacement stays small. This
    // DLL sits far outside the +/-2GB window a disp32 can address.
    constexpr uint32_t kSampleCounterOffset = 0x2F0;
    constexpr uint32_t kObjectSlotOffset = 0x300;
    constexpr uint32_t kInnerSlotOffset = 0x308;

    // Second hook: the emitter entry point runs during scene setup and hands the
    // renderer its camera object pointer as the third argument.
    constexpr uint8_t kEmitterSignature[] = {
      0x48, 0x8B, 0xC4, 0x48, 0x89, 0x58, 0x18, 0x48, 0x89,
      0x70, 0x20, 0x48, 0x89, 0x48, 0x08, 0x55, 0x57, 0x41, 0x56,
    };

    constexpr uint8_t kEmitterPrologue[] = {
      0x48, 0x8B, 0xC4,             // mov rax, rsp
      0x48, 0x89, 0x58, 0x18,       // mov [rax+18h], rbx
      0x48, 0x89, 0x70, 0x20,       // mov [rax+20h], rsi
      0x48, 0x89, 0x48, 0x08,       // mov [rax+08h], rcx
    };

    constexpr uint32_t kCameraObjectSlotOffset = 0x300;

    // Filled by the vertex batch probe; declared here because the snapshot
    // accessor reads it.
    constexpr uint32_t kTransformArgsOffset = 0x320;
    // Four camera arguments, newest first, so a frame that batches several
    // cameras can be told apart from one that batches a single camera.
    constexpr uint32_t kCameraHistoryOffset = 0x3E0;
    uint8_t* g_pTransformDetour = nullptr;
    uint8_t* g_pTransformTrampoline = nullptr;

    uint8_t* g_pTrampoline = nullptr;
    uint8_t* g_pDetour = nullptr;
    uint8_t* g_pEmitterTrampoline = nullptr;
    uint8_t* g_pEmitterDetour = nullptr;
    bool g_installed = false;

    void* findPattern(uint8_t* pStart, size_t size) {
      if (size < sizeof(kSignature))
        return nullptr;
      const size_t limit = size - sizeof(kSignature);
      for (size_t i = 0; i < limit; ++i) {
        if (pStart[i] != kSignature[0])
          continue;
        if (std::memcmp(pStart + i, kSignature, sizeof(kSignature)) == 0)
          return pStart + i;
      }
      return nullptr;
    }

    void emit(uint8_t*& pCursor, std::initializer_list<uint8_t> bytes) {
      for (uint8_t value : bytes)
        *pCursor++ = value;
    }

    void emitU64(uint8_t*& pCursor, uint64_t value) {
      std::memcpy(pCursor, &value, sizeof(value));
      pCursor += sizeof(value);
    }

    void emitDisplacement(uint8_t* pCursor, uint8_t* pDestination) {
      // The cursor points at the displacement of an instruction that ends four
      // bytes later.
      const int64_t delta = pDestination - (pCursor + sizeof(int32_t));
      const auto displacement = static_cast<int32_t>(delta);
      std::memcpy(pCursor, &displacement, sizeof(displacement));
    }

    void emitStoreFrame(uint8_t*& pCursor, uint8_t* pDestination) {
      emit(pCursor, { 0x48, 0x89, 0x05 });               // mov [rip+disp32], rax
      emitDisplacement(pCursor, pDestination);
      pCursor += sizeof(int32_t);
    }

  }

  namespace {

    // Installs the emitter hook, which records the camera object pointer that the
    // renderer passes as its third argument. The pointer is stored verbatim; no
    // object memory is read here.
    bool installEmitterHook(uint8_t* pBase, size_t imageSize) {
      if (g_pEmitterDetour)
        return true;

      constexpr size_t kSigSize = sizeof(kEmitterSignature);
      const size_t limit = imageSize > kSigSize ? imageSize - kSigSize : 0;
      uint8_t* pTarget = nullptr;
      for (size_t i = 0; i < limit; ++i) {
        if (pBase[i] != kEmitterSignature[0])
          continue;
        if (std::memcmp(pBase + i, kEmitterSignature, kSigSize) == 0) {
          pTarget = pBase + i;
          break;
        }
      }
      if (!pTarget) {
        log("ACZ camera hook: emitter signature not found");
        return false;
      }

      constexpr size_t kSize = sizeof(kEmitterPrologue);
      g_pEmitterTrampoline = static_cast<uint8_t*>(VirtualAlloc(
        nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
      g_pEmitterDetour = static_cast<uint8_t*>(VirtualAlloc(
        nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
      if (!g_pEmitterTrampoline || !g_pEmitterDetour)
        return false;
      std::memset(g_pEmitterDetour, 0, 4096);

      std::memcpy(g_pEmitterTrampoline, pTarget, kSize);
      uint8_t* pTrampolineCursor = g_pEmitterTrampoline + kSize;
      emit(pTrampolineCursor, { 0xFF, 0x25, 0x00, 0x00, 0x00, 0x00 });
      emitU64(pTrampolineCursor, reinterpret_cast<uint64_t>(pTarget + kSize));

      uint8_t* pCursor = g_pEmitterDetour;
      emit(pCursor, { 0x50 });                           // push rax
      emit(pCursor, { 0x48, 0x83, 0xEC, 0x10 });         // sub rsp, 10h
      emit(pCursor, { 0x0F, 0x11, 0x04, 0x24 });         // movups [rsp], xmm0

      emit(pCursor, { 0x48, 0xFF, 0x05 });               // inc qword [rip+disp32]
      emitDisplacement(pCursor, g_pEmitterDetour + kSampleCounterOffset);
      pCursor += sizeof(int32_t);

      emit(pCursor, { 0x4D, 0x85, 0xC0 });               // test r8, r8
      uint8_t* pNullPatch = pCursor;
      emit(pCursor, { 0x74, 0x00 });                     // jz done

      emit(pCursor, { 0x49, 0x8B, 0x80 });               // mov rax, [r8+0xA8]
      emit(pCursor, { 0xA8, 0x00, 0x00, 0x00 });
      emitStoreFrame(pCursor, g_pEmitterDetour + kCameraObjectSlotOffset);

      uint8_t* pDone = pCursor;
      pNullPatch[1] = static_cast<uint8_t>(pDone - (pNullPatch + 2));

      emit(pCursor, { 0x0F, 0x10, 0x04, 0x24 });         // movups xmm0, [rsp]
      emit(pCursor, { 0x48, 0x83, 0xC4, 0x10 });         // add rsp, 10h
      emit(pCursor, { 0x58 });                           // pop rax
      emit(pCursor, { 0xFF, 0x25, 0x00, 0x00, 0x00, 0x00 });
      emitU64(pCursor, reinterpret_cast<uint64_t>(g_pEmitterTrampoline));

      DWORD protection = 0;
      if (!VirtualProtect(pTarget, kSize, PAGE_EXECUTE_READWRITE, &protection))
        return false;

      uint8_t patch[24] = { 0xFF, 0x25, 0x00, 0x00, 0x00, 0x00 };
      std::memcpy(patch + 6, &g_pEmitterDetour, sizeof(uint64_t));
      for (size_t i = 14; i < kSize; ++i)
        patch[i] = 0x90;
      std::memcpy(pTarget, patch, kSize);

      DWORD ignored = 0;
      VirtualProtect(pTarget, kSize, protection, &ignored);
      FlushInstructionCache(GetCurrentProcess(), pTarget, kSize);
      FlushInstructionCache(GetCurrentProcess(), g_pEmitterDetour, 4096);

      log("ACZ camera hook: emitter hooked at 0x%llx detour=%p",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(pTarget)),
        g_pEmitterDetour);
      return true;
    }

  }

  bool readAczCameraProbe(AczCameraProbe& out) {
    if (!g_installed || !g_pDetour)
      return false;

    const auto* pCounter = reinterpret_cast<const long*>(g_pDetour + kSampleCounterOffset);
    if (*pCounter == 0)
      return false;

    out.samples = static_cast<uint32_t>(*pCounter);
    std::memcpy(&out.object, g_pDetour + kObjectSlotOffset, sizeof(out.object));
    std::memcpy(&out.inner, g_pDetour + kInnerSlotOffset, sizeof(out.inner));
    out.camera = 0;
    if (g_pTransformDetour)
      std::memcpy(out.transformArgs, g_pTransformDetour + kTransformArgsOffset, sizeof(out.transformArgs));
    if (g_pEmitterDetour)
      std::memcpy(&out.camera, g_pEmitterDetour + kCameraObjectSlotOffset, sizeof(out.camera));
    return true;
  }

  uint64_t readAczTransformSamples() {
    if (!g_pTransformDetour)
      return 0;
    const auto* pCounter = reinterpret_cast<const volatile uint64_t*>(
      g_pTransformDetour + kSampleCounterOffset);
    return *pCounter;
  }

  uint32_t readAczTransformCameras(uint64_t out[4]) {
    if (!g_pTransformDetour)
      return 0;
    uint32_t populated = 0;
    for (uint32_t i = 0; i < 4; ++i) {
      std::memcpy(&out[i], g_pTransformDetour + kCameraHistoryOffset + i * 8,
        sizeof(uint64_t));
      if (out[i])
        ++populated;
    }
    return populated;
  }

  namespace {

    // Probe for the vertex batch transform. Only the four arguments are stored, so
    // a wrong candidate cannot fault.
    constexpr uint8_t kTransformSignature[] = {
      0x40, 0x53, 0x56, 0x57, 0x41, 0x54, 0x41, 0x55, 0x41,
      0x56, 0x41, 0x57, 0x48, 0x81, 0xEC, 0x80, 0x02, 0x00, 0x00,
    };

    constexpr uint8_t kTransformPrologue[] = {
      0x40, 0x53,                         // push rbx
      0x56,                               // push rsi
      0x57,                               // push rdi
      0x41, 0x54,                         // push r12
      0x41, 0x55,                         // push r13
      0x41, 0x56,                         // push r14
      0x41, 0x57,                         // push r15
      0x48, 0x81, 0xEC, 0x80, 0x02, 0x00, 0x00, // sub rsp, 280h
    };



    bool installTransformProbe(uint8_t* pBase, size_t imageSize) {
      if (g_pTransformDetour)
        return true;

      constexpr size_t kSigSize = sizeof(kTransformSignature);
      const size_t limit = imageSize > kSigSize ? imageSize - kSigSize : 0;
      uint8_t* pTarget = nullptr;
      for (size_t i = 0; i < limit; ++i) {
        if (pBase[i] != kTransformSignature[0])
          continue;
        if (std::memcmp(pBase + i, kTransformSignature, kSigSize) == 0) {
          pTarget = pBase + i;
          break;
        }
      }
      if (!pTarget) {
        log("ACZ transform probe: signature not found");
        return false;
      }

      constexpr size_t kSize = sizeof(kTransformPrologue);
      g_pTransformTrampoline = static_cast<uint8_t*>(VirtualAlloc(
        nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
      g_pTransformDetour = static_cast<uint8_t*>(VirtualAlloc(
        nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
      if (!g_pTransformTrampoline || !g_pTransformDetour)
        return false;
      std::memset(g_pTransformDetour, 0, 4096);

      std::memcpy(g_pTransformTrampoline, pTarget, kSize);
      uint8_t* pTrampolineCursor = g_pTransformTrampoline + kSize;
      emit(pTrampolineCursor, { 0xFF, 0x25, 0x00, 0x00, 0x00, 0x00 });
      emitU64(pTrampolineCursor, reinterpret_cast<uint64_t>(pTarget + kSize));

      uint8_t* pCursor = g_pTransformDetour;
      emit(pCursor, { 0x50 });                             // push rax
      emit(pCursor, { 0x48, 0x83, 0xEC, 0x20 });           // sub rsp, 20h
      emit(pCursor, { 0x0F, 0x11, 0x04, 0x24 });           // movups [rsp], xmm0

      const struct { uint8_t opcode[3]; int reg; uint32_t slot; } stores[] = {
        { { 0x48, 0x89, 0x0D }, 0, 0x00 },                 // rcx
        { { 0x48, 0x89, 0x15 }, 1, 0x08 },                 // rdx
        { { 0x4C, 0x89, 0x05 }, 2, 0x10 },                 // r8
        { { 0x4C, 0x89, 0x0D }, 3, 0x18 },                 // r9
      };
      for (const auto& store : stores) {
        emit(pCursor, { store.opcode[0], store.opcode[1], store.opcode[2] });
        emitDisplacement(pCursor, g_pTransformDetour + kTransformArgsOffset + store.slot);
        pCursor += sizeof(int32_t);
      }

      // Shift a four entry history of the camera argument so every batch call in
      // a frame stays observable, not only the last one.
      for (uint32_t slot : { 0x3F0u, 0x3E8u, 0x3E0u }) {
        emit(pCursor, { 0x48, 0x8B, 0x05 });               // mov rax, [rip+disp32]
        emitDisplacement(pCursor, g_pTransformDetour + slot);
        pCursor += sizeof(int32_t);
        emitStoreFrame(pCursor, g_pTransformDetour + slot + 8);
      }
      emit(pCursor, { 0x4C, 0x89, 0x05 });                 // mov [rip+disp32], r8
      emitDisplacement(pCursor, g_pTransformDetour + kCameraHistoryOffset);
      pCursor += sizeof(int32_t);

      emit(pCursor, { 0x48, 0xFF, 0x05 });                 // inc qword [rip+disp32]
      emitDisplacement(pCursor, g_pTransformDetour + kSampleCounterOffset);
      pCursor += sizeof(int32_t);

      emit(pCursor, { 0x0F, 0x10, 0x04, 0x24 });           // movups xmm0, [rsp]
      emit(pCursor, { 0x48, 0x83, 0xC4, 0x20 });           // add rsp, 20h
      emit(pCursor, { 0x58 });                             // pop rax
      emit(pCursor, { 0xFF, 0x25, 0x00, 0x00, 0x00, 0x00 });
      emitU64(pCursor, reinterpret_cast<uint64_t>(g_pTransformTrampoline));

      DWORD protection = 0;
      if (!VirtualProtect(pTarget, kSize, PAGE_EXECUTE_READWRITE, &protection))
        return false;

      uint8_t patch[24] = { 0xFF, 0x25, 0x00, 0x00, 0x00, 0x00 };
      std::memcpy(patch + 6, &g_pTransformDetour, sizeof(uint64_t));
      for (size_t i = 14; i < kSize; ++i)
        patch[i] = 0x90;
      std::memcpy(pTarget, patch, kSize);

      DWORD ignored = 0;
      VirtualProtect(pTarget, kSize, protection, &ignored);
      FlushInstructionCache(GetCurrentProcess(), pTarget, kSize);
      FlushInstructionCache(GetCurrentProcess(), g_pTransformDetour, 4096);

      log("ACZ transform probe: hooked at 0x%llx detour=%p",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(pTarget)),
        g_pTransformDetour);
      return true;
    }

  }

  bool installAczCameraHook() {
    if (g_installed)
      return true;

    HMODULE pExe = GetModuleHandleW(nullptr);
    if (!pExe) {
      log("ACZ camera hook: no executable module");
      return false;
    }

    auto* pBase = reinterpret_cast<uint8_t*>(pExe);
    auto* pDos = reinterpret_cast<IMAGE_DOS_HEADER*>(pBase);
    auto* pNt = reinterpret_cast<IMAGE_NT_HEADERS*>(pBase + pDos->e_lfanew);
    const size_t imageSize = pNt->OptionalHeader.SizeOfImage;

    void* pHookTarget = findPattern(pBase, imageSize);
    if (!pHookTarget) {
      log("ACZ camera hook: signature not found");
      return false;
    }

    auto* pTarget = static_cast<uint8_t*>(pHookTarget);
    constexpr size_t kPrologueSize = sizeof(kPrologue);

    g_pTrampoline = static_cast<uint8_t*>(VirtualAlloc(
      nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
    g_pDetour = static_cast<uint8_t*>(VirtualAlloc(
      nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
    if (!g_pTrampoline || !g_pDetour) {
      log("ACZ camera hook: allocation failed");
      return false;
    }

    std::memset(g_pDetour, 0, 4096);

    std::memcpy(g_pTrampoline, pTarget, kPrologueSize);
    uint8_t* pTrampolineCursor = g_pTrampoline + kPrologueSize;
    emit(pTrampolineCursor, { 0xFF, 0x25, 0x00, 0x00, 0x00, 0x00 });
    emitU64(pTrampolineCursor, reinterpret_cast<uint64_t>(pTarget + kPrologueSize));

    uint8_t* pDetourCursor = g_pDetour;
    emit(pDetourCursor, { 0x50 });                       // push rax
    emit(pDetourCursor, { 0x48, 0x83, 0xEC, 0x10 });     // sub rsp, 10h
    emit(pDetourCursor, { 0x0F, 0x11, 0x04, 0x24 });     // movups [rsp], xmm0

    emit(pDetourCursor, { 0x48, 0xFF, 0x05 });           // inc qword [rip+disp32]
    emitDisplacement(pDetourCursor, g_pDetour + kSampleCounterOffset);
    pDetourCursor += sizeof(int32_t);

    emit(pDetourCursor, { 0x48, 0x89, 0x0D });           // mov [rip+disp32], rcx
    emitDisplacement(pDetourCursor, g_pDetour + kObjectSlotOffset);
    pDetourCursor += sizeof(int32_t);

    emit(pDetourCursor, { 0x48, 0x85, 0xC9 });           // test rcx, rcx
    uint8_t* pNullPatch = pDetourCursor;
    emit(pDetourCursor, { 0x74, 0x00 });                 // jz done

    emit(pDetourCursor, { 0x48, 0x8B, 0x41, 0x08 });     // mov rax, [rcx+8]
    emitStoreFrame(pDetourCursor, g_pDetour + kInnerSlotOffset);

    uint8_t* pDone = pDetourCursor;
    pNullPatch[1] = static_cast<uint8_t>(pDone - (pNullPatch + 2));

    emit(pDetourCursor, { 0x0F, 0x10, 0x04, 0x24 });     // movups xmm0, [rsp]
    emit(pDetourCursor, { 0x48, 0x83, 0xC4, 0x10 });     // add rsp, 10h
    emit(pDetourCursor, { 0x58 });                       // pop rax
    emit(pDetourCursor, { 0xFF, 0x25, 0x00, 0x00, 0x00, 0x00 });
    emitU64(pDetourCursor, reinterpret_cast<uint64_t>(g_pTrampoline));

    DWORD protection = 0;
    if (!VirtualProtect(pTarget, kPrologueSize, PAGE_EXECUTE_READWRITE, &protection)) {
      log("ACZ camera hook: VirtualProtect failed error=%lu", GetLastError());
      return false;
    }

    uint8_t patch[16] = { 0xFF, 0x25, 0x00, 0x00, 0x00, 0x00 };
    std::memcpy(patch + 6, &g_pDetour, sizeof(uint64_t));
    for (size_t i = 14; i < kPrologueSize; ++i)
      patch[i] = 0x90;
    std::memcpy(pTarget, patch, kPrologueSize);

    DWORD ignored = 0;
    VirtualProtect(pTarget, kPrologueSize, protection, &ignored);
    FlushInstructionCache(GetCurrentProcess(), pTarget, kPrologueSize);
    FlushInstructionCache(GetCurrentProcess(), g_pDetour, 4096);

    installEmitterHook(pBase, imageSize);
    installTransformProbe(pBase, imageSize);

    g_installed = true;
    log("ACZ camera hook: installed target=0x%llx detour=%p trampoline=%p",
      static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(pTarget)),
      g_pDetour, g_pTrampoline);
    return true;
  }

}
