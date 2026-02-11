# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for 동전게임 공략 가이드
"""

import os
import sys

block_cipher = None

# 프로젝트 루트
PROJECT_ROOT = os.path.dirname(os.path.abspath(SPEC))

a = Analysis(
    [os.path.join(PROJECT_ROOT, 'main.py')],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=[
        (os.path.join(PROJECT_ROOT, 'config.py'), '.'),
        (os.path.join(PROJECT_ROOT, 'api.txt'), '.'),
    ],
    hiddenimports=[
        'models',
        'models.coin',
        'vision',
        'vision.screen_capture',
        'vision.gemini_analyzer',
        'physics',
        'physics.engine',
        'physics.simulator',
        'solver',
        'solver.optimizer',
        'solver.strategy',
        'ui',
        'ui.overlay',
        'utils',
        'utils.coordinate_mapper',
        'utils.state_detector',
        'pymunk',
        'pymunk.autogeometry',
        'pymunk.constraint',
        'pymunk.body',
        'pymunk.shape',
        'pymunk.space',
        'mss',
        'mss.windows',
        'pygetwindow',
        'pygetwindow._pygetwindow_win',
        'PIL',
        'PIL.Image',
        'numpy',
        'cv2',
        'google.genai',
        'google.generativeai',
        'tkinter',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
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
    icon=None,
)
