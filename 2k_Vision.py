'''


WELCOME TO


 ___  _  __  __     ___     _             
|__ \| |/ / \ \   / (_)___(_) ___  _ __  
  ) | ' /   \ \ / /| / __| |/ _ \| '_ \ 
 / /| . \    \ V / | \__ \ | (_) | | | |
|___|_|\_\    \_/  |_|___/_|\___/|_| |_|


'''
import os, ctypes, requests

class GCVWorker:
    def __init__(s, w, h):
        _script_dir = os.path.dirname(__file__)
        _f = 'ch.dll'
        _p = os.path.join(_script_dir, 'bin', _f)
        if not os.path.exists(_p):
            _r = requests.get(f'https://2kv.inputsense.com/nba2k/bin/{_f}', timeout=6)
            _r.raise_for_status()
            os.makedirs(os.path.dirname(_p), exist_ok=True)
            with open(_p, 'wb') as _o:
                _o.write(_r.content)
        s._d = ctypes.PyDLL(_p)
        s._z = bytearray(1)
        s._d.r.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
        s._d.r.restype = ctypes.c_int
        s._d.p.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        s._d.p.restype = ctypes.c_int
        s._d.g.argtypes = []
        s._d.g.restype = ctypes.c_void_p
        s._d.x.argtypes = []
        s._d.x.restype = None
        s._p = s._d.p
        s._g = s._d.g
        s._x = s._d.x
        rc = s._d.r(w, h, _script_dir.encode('utf-8'))
        if rc != 0:
            raise Exception(f'DLL init failed with code {rc}')

    def close(s):
        _x = getattr(s, '_x', None)
        if _x is not None:
            s._x = None
            _x()

    def __del__(s):
        s.close()

    def process(s, frame):
        sz = s._p(frame.ctypes.data, frame.shape[1], frame.shape[0], frame.strides[0])
        if sz <= 0:
            return frame, s._z
        _result = bytearray(sz)
        ctypes.memmove((ctypes.c_ubyte * sz).from_buffer(_result), s._g(), sz)
        return frame, _result
