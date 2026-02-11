# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[('config.py', '.'), ('api.txt', '.')],
    hiddenimports=['models', 'models.coin', 'vision', 'vision.screen_capture', 'vision.gemini_analyzer', 'physics', 'physics.engine', 'physics.simulator', 'solver', 'solver.optimizer', 'solver.strategy', 'ui', 'ui.overlay', 'utils', 'utils.coordinate_mapper', 'utils.state_detector', 'pymunk', 'mss', 'mss.windows', 'pygetwindow', 'google.generativeai', 'PIL', 'PIL.Image', 'numpy', 'cv2'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'torchaudio', 'PyQt6', 'PyQt5', 'PySide6', 'PySide2', 'matplotlib', 'scipy', 'pandas', 'IPython', 'jupyter', 'notebook', 'transformers', 'tensorflow', 'keras', 'sklearn', 'scikit-learn', 'sympy', 'wx', 'PyQt6.QtWebEngineWidgets', 'fugashi', 'websockets'],
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
    name='CoinGameAssistant',
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
