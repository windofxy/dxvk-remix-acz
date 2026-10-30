subdir('usd-plugins')
subdir('tracy')
subdir('util')
subdir('spirv')
subdir('vulkan')
subdir('lssusd')
subdir('dxvk')

if get_option('enable_d3d12')
  subdir('d3d12')
endif

if get_option('enable_dxgi_proxy')
  if get_option('enable_dxgi')
    error('The ACZ DXGI proxy cannot be built together with the DXVK DXGI frontend.')
  endif
  subdir('dxgi_proxy')
endif

if get_option('enable_dxgi')
  if not get_option('enable_d3d11')
    error('D3D11 is required for DXGI.')
  endif
  subdir('dxgi')
endif

if get_option('enable_d3d10') or get_option('enable_d3d11') or get_option('enable_tests')
  subdir('dxbc')
endif

if get_option('enable_d3d11')
  subdir('d3d11')
endif

if get_option('enable_d3d10')
  if not get_option('enable_d3d11')
    error('D3D11 is required for D3D10.')
  endif
  subdir('d3d10')
endif

if get_option('enable_d3d9')
  subdir('dxso')
  subdir('d3d9')
endif

# Nothing selected
if not get_option('enable_d3d9') and not get_option('enable_d3d10') and not get_option('enable_d3d11') and not get_option('enable_tests')
  warning('Nothing selected to be built. Are you missing a frontend or tests?')
endif
