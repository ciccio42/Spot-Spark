import argparse
import sys

import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.util
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient

g_image_click = None
g_image_display = None
WINDOW = 'Click to grasp'


def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        h, w = clone.shape[:2]
        cv2.line(clone, (0, y), (w, y), (30, 30, 30), 2)
        cv2.line(clone, (x, 0), (x, h), (30, 30, 30), 2)
        cv2.imshow(WINDOW, clone)


def get_image(image_client, source):
    responses = image_client.get_image_from_sources([source])
    if len(responses) != 1:
        raise RuntimeError(f'Numero di immagini non valido: {len(responses)}')
    resp = responses[0]
    img_pb = resp.shot.image
    dtype = np.uint16 if img_pb.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16 else np.uint8
    img = np.frombuffer(img_pb.data, dtype=dtype)
    if img_pb.format == image_pb2.Image.FORMAT_RAW:
        img = img.reshape(img_pb.rows, img_pb.cols)
    else:
        img = cv2.imdecode(img, -1)
    return resp, img


def print_grasp_info(resp, click):
    snapshot = resp.shot.transforms_snapshot
    print('\n===== INFO PER IL GRASP =====')
    print(f'Pixel cliccato (x, y): {click}')
    print(f'frame_name_image_sensor: {resp.shot.frame_name_image_sensor}')
    print(f'Sorgente: {resp.source.name}')
    print(f'Risoluzione: {resp.shot.image.cols}x{resp.shot.image.rows}')

    print('\n--- Camera model (pinhole) ---')
    print(resp.source.pinhole)

    print('--- Transforms snapshot (child -> parent) ---')
    for child, edge in snapshot.child_to_parent_edge_map.items():
        p = edge.parent_tform_child.position
        q = edge.parent_tform_child.rotation
        print(f'{child:30s} -> {edge.parent_frame_name:20s} '
              f'pos=({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) '
              f'quat=(w={q.w:.3f}, x={q.x:.3f}, y={q.y:.3f}, z={q.z:.3f})')

    print('\nFrame ids nello snapshot:', sorted(snapshot.child_to_parent_edge_map.keys()))


def main():
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-i', '--image-source', default='hand_color_image')
    config = parser.parse_args()

    sdk = bosdyn.client.create_standard_sdk('ImageClickClient')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)

    image_client = robot.ensure_client(ImageClient.default_service_name)
    resp, img = get_image(image_client, config.image_source)

    global g_image_display
    g_image_display = img
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, cv_mouse_callback)
    cv2.imshow(WINDOW, g_image_display)

    while g_image_click is None:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            print('"q" premuto, esco.')
            cv2.destroyAllWindows()
            return True

    cv2.destroyAllWindows()
    print_grasp_info(resp, g_image_click)
    return True


if __name__ == '__main__':
    try:
        ok = main()
    except Exception:
        bosdyn.client.util.get_logger().exception('Eccezione')
        ok = False
    sys.exit(0 if ok else 1)