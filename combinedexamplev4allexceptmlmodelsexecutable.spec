# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['combinedexamplev4allexceptmlmodelsexecutable.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\kevin\\.cache\\whisper', 'whisper'), ('c:\\code\\mediapipeGazeTrackingNewLogic\\gaze_tracking.py', 'gaze_tracking'), ('C:\\code\\mediapipeGazeTrackingNewLogic\\env\\Lib\\site-packages\\mediapipe\\modules\\face_landmark', 'mediapipe/modules/face_landmark')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='combinedexamplev4allexceptmlmodelsexecutable',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
