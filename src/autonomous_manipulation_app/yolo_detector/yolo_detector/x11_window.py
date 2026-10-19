"""Keep an X11 window always on top, through the window manager (EWMH _NET_WM_STATE_ABOVE).

cv2.WND_PROP_TOPMOST is accepted but ignored by the OpenCV Qt backend, so the request is sent
directly to the window manager, as `wmctrl -r <title> -b add,above` does. Only libX11 is needed
(loaded with ctypes, always present with an X11 GUI backend): no extra package.
"""
import ctypes
import ctypes.util

_NET_WM_STATE_ADD = 1
_SOURCE_APPLICATION = 1
_CLIENT_MESSAGE = 33
_SUBSTRUCTURE_REDIRECT_MASK = 1 << 20
_SUBSTRUCTURE_NOTIFY_MASK = 1 << 19


class _XClientMessageEvent(ctypes.Structure):
    _fields_ = [
        ('type', ctypes.c_int),
        ('serial', ctypes.c_ulong),
        ('send_event', ctypes.c_int),
        ('display', ctypes.c_void_p),
        ('window', ctypes.c_ulong),
        ('message_type', ctypes.c_ulong),
        ('format', ctypes.c_int),
        ('data', ctypes.c_long * 5),
    ]


class _XEvent(ctypes.Union):
    # XEvent is a union padded to 24 longs
    _fields_ = [('xclient', _XClientMessageEvent), ('pad', ctypes.c_long * 24)]


def _load_xlib():
    path = ctypes.util.find_library('X11') or 'libX11.so.6'
    xlib = ctypes.cdll.LoadLibrary(path)
    xlib.XOpenDisplay.restype = ctypes.c_void_p
    xlib.XOpenDisplay.argtypes = [ctypes.c_char_p]
    xlib.XDefaultRootWindow.restype = ctypes.c_ulong
    xlib.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    xlib.XInternAtom.restype = ctypes.c_ulong
    xlib.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    xlib.XQueryTree.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
                                ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
                                ctypes.POINTER(ctypes.c_uint)]
    xlib.XFetchName.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_char_p)]
    xlib.XFree.argtypes = [ctypes.c_void_p]
    xlib.XSendEvent.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_long,
                                ctypes.POINTER(_XEvent)]
    xlib.XFlush.argtypes = [ctypes.c_void_p]
    xlib.XCloseDisplay.argtypes = [ctypes.c_void_p]
    return xlib


def _find_windows(xlib, display, window, title, found):
    """Depth-first search of the windows whose WM_NAME is title."""
    name = ctypes.c_char_p()
    if xlib.XFetchName(display, window, ctypes.byref(name)) and name.value is not None:
        if name.value.decode(errors='ignore') == title:
            found.append(window)
        xlib.XFree(name)
    root, parent = ctypes.c_ulong(), ctypes.c_ulong()
    children, count = ctypes.POINTER(ctypes.c_ulong)(), ctypes.c_uint()
    if xlib.XQueryTree(display, window, ctypes.byref(root), ctypes.byref(parent),
                       ctypes.byref(children), ctypes.byref(count)):
        for i in range(count.value):
            _find_windows(xlib, display, children[i], title, found)
        if children:
            xlib.XFree(children)


def set_always_on_top(title):
    """Asks the window manager to keep the window(s) titled `title` above the others.
    Returns True if at least one window was found (the window must be already shown)."""
    xlib = _load_xlib()
    display = xlib.XOpenDisplay(None)
    if not display:
        return False
    try:
        root = xlib.XDefaultRootWindow(display)
        windows = []
        _find_windows(xlib, display, root, title, windows)
        wm_state = xlib.XInternAtom(display, b'_NET_WM_STATE', False)
        above = xlib.XInternAtom(display, b'_NET_WM_STATE_ABOVE', False)
        for window in windows:
            event = _XEvent()
            event.xclient.type = _CLIENT_MESSAGE
            event.xclient.send_event = 1
            event.xclient.window = window
            event.xclient.message_type = wm_state
            event.xclient.format = 32
            event.xclient.data[:] = [_NET_WM_STATE_ADD, above, 0, _SOURCE_APPLICATION, 0]
            xlib.XSendEvent(display, root, False,
                            _SUBSTRUCTURE_REDIRECT_MASK | _SUBSTRUCTURE_NOTIFY_MASK, ctypes.byref(event))
        xlib.XFlush(display)
        return bool(windows)
    finally:
        xlib.XCloseDisplay(display)
