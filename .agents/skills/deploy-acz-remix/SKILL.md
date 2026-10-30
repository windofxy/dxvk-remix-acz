---
name: deploy-acz-remix
description: Deploy and verify the native x64 ACZ D3D12 frontend with RTX Remix runtime dependencies.
---

# Deploy ACZ Remix

Use this skill for `acz.exe`, not for a normal D3D9 Remix game.

The ACZ executable is native x64 D3D12. Its entry point is the adapted
`d3d12.dll` beside `acz.exe`; the x64 Remix package's `d3d9.dll` shim and
`d3d9_x64.dll` implementation are a separate D3D9 deployment model.

## Build

From the repository root:

```powershell
meson compile -C _CompRelease_acz_frontend_x64 d3d12 dxgi
```

Use the release frontend for game testing. Expected outputs are
`src/d3d12/d3d12.dll` and, when the diagnostic proxy is enabled,
`src/dxgi_proxy/dxgi.dll`.

## Deploy

1. Stop `acz.exe`.
2. Copy the built ACZ `d3d12.dll` to the executable directory.
3. Copy the optional custom `dxgi.dll` to that directory.
4. Copy the root DLLs from `rtx-remix-for-x64-games-release-remix-1.6.1`, preserving names.
5. Copy the package `usd/` directory recursively.
6. Keep required project-built dependencies such as `NRC_D3D12.dll` beside the executable.
7. Do not use `d3d9.dll` as a replacement for the ACZ `d3d12.dll`.

The package's D3D9 files are not needed by the native D3D12 path. Do not copy
Windows system DLLs or NVIDIA driver DLLs into the game directory without
loader evidence.

For repeatable Meson installs, define an `ACZero` entry in `gametargets.conf`
whose `outputdir` is the directory containing `acz.exe`, then run:

```powershell
meson setup --reconfigure _CompRelease_acz_frontend_x64
meson compile -C _CompRelease_acz_frontend_x64
meson install -C _CompRelease_acz_frontend_x64 --tags ACZero --only-changed
```

Inspect the install manifest first. Manual deployment is preferred if a tagged
install would overwrite D3D9 test files.

## Verify

Delete stale `remix-d3d12-proxy.log`, `remix-dxgi-proxy.log`,
`remix-d3d12-remix.log`, `remix-d3d12-bindings.csv`, and `metrics.txt` before each
new DLL test. A valid run must show successful D3D12 device/swapchain creation
and successful `Present`. Geometry bridge records prove capture only;
`D3D12 native Remix scene submit` and scene state records are needed to prove
submission to the Remix scene manager.

See `Docs/d3d12-compat-build-deploy.md` for the full file layout and commands, and
`Docs/d3d12-compat-architecture.md` for the runtime data flow.
