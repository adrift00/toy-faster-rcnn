import numpy as np


def delta2bbox(src_bbox, delta):
    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + src_height * 0.5
    src_ctr_x = src_bbox[:, 1] + src_width * 0.5

    dy = delta[:, 0::4]
    dx = delta[:, 1::4]
    dh = delta[:, 2::4]
    dw = delta[:, 3::4]

    dst_ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    dst_ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    dst_height = src_height[:, np.newaxis] * np.exp(dh)
    dst_width = src_width[:, np.newaxis] * np.exp(dw)


    dst_boxes = np.zeros_like(delta)
    dst_boxes[:, 0::4] = dst_ctr_y - dst_height * 0.5
    dst_boxes[:, 1::4] = dst_ctr_x - dst_width * 0.5
    dst_boxes[:, 2::4] = dst_ctr_y + dst_height * 0.5
    dst_boxes[:, 3::4] = dst_ctr_x + dst_width * 0.5

    return dst_boxes


def bbox2delta(src_bbox, dst_bbox):
    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width= src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + src_height * 0.5
    src_ctr_x = src_bbox[:, 1] + src_width * 0.5

    dst_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    dst_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    dst_ctr_y = dst_bbox[:, 0] + dst_height * 0.5
    dst_ctr_x = dst_bbox[:, 1] + dst_width * 0.5

    # note: eps is needed to avoid divide 0
    eps = np.finfo(src_height.dtype).eps
    src_height = np.maximum(src_height, eps)
    src_width = np.maximum(src_width, eps)

    dy = (dst_ctr_y - src_ctr_y) / src_height
    dx = (dst_ctr_x - src_ctr_x) / src_width
    dh = np.log(dst_height / src_height)
    dw = np.log(dst_width / src_width)


    delta = np.vstack((dy, dx, dh, dw)).transpose()
    return delta

# TODO: the function is copied from simple_faster_rcnn, understand it soon.
def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.

    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)
