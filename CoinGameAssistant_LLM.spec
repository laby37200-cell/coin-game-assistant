# -*- mode: python ; coding: utf-8 -*-
# LLM-Only 버전 — 물리엔진/torch 없이 Gemini만 사용

from PyInstaller.utils.hooks import collect_all, collect_submodules

# google.genai 패키지의 모든 하위 모듈/데이터 수집
_genai_datas, _genai_binaries, _genai_hiddenimports = collect_all('google.genai')
_genai_hiddenimports += collect_submodules('google.auth')
_genai_hiddenimports += collect_submodules('google.api_core')
_genai_hiddenimports += collect_submodules('httpx')

a = Analysis(
    ['main_llm.py'],
    pathex=['.'],
    binaries=_genai_binaries,
    datas=[
        ('config.py', '.'),
        ('api.txt', '.'),
        ('knowledge', 'knowledge'),
    ] + _genai_datas,
    hiddenimports=[
        'models', 'models.coin',
        'vision', 'vision.screen_capture',
        'solver', 'solver.llm_advisor',
        'ui', 'ui.overlay', 'ui.control_panel', 'ui.log_windows',
        'utils', 'utils.state_detector',
        'mss', 'mss.windows',
        'pygetwindow', 'pygetwindow._pygetwindow_win',
        'PIL', 'PIL.Image',
        'numpy', 'cv2', 'tkinter',
    ] + _genai_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyi_rth_utf8.py'],
    excludes=[
        'torch', 'torchvision', 'torchaudio',
        'PyQt6', 'PyQt5', 'PySide6', 'PySide2',
        'matplotlib', 'scipy', 'pandas',
        'IPython', 'jupyter', 'notebook',
        'transformers', 'tensorflow', 'keras',
        'sklearn', 'scikit-learn', 'sympy', 'wx',
        'pymunk', 'fugashi',
        'physics', 'ai',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='CoinGameAssistant_LLM',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
