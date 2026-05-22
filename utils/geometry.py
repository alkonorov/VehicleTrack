import numpy as np
def iou(box_a, box_b) -> float:
    """
    IoU между двумя bbox
    """

    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    inter_area = max(0, inter_x2-inter_x1) * max(0, inter_y2-inter_y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0
    return inter_area / union



def iou_matrix(boxes_a, boxes_b) -> np.ndarray:
    """
    Матрица IoU [len(a) x len(b)]
    Args:
        boxes_a: np.ndarray shape (N, 4)
        boxes_b: np.ndarray shape (M, 4)
    """
    # Разворачиваем координаты
    a_x1 = boxes_a[:, 0, np.newaxis]  # (N, 1)
    a_y1 = boxes_a[:, 1, np.newaxis]
    a_x2 = boxes_a[:, 2, np.newaxis]
    a_y2 = boxes_a[:, 3, np.newaxis]

    b_x1 = boxes_b[:, 0]  # (M,)
    b_y1 = boxes_b[:, 1]
    b_x2 = boxes_b[:, 2]
    b_y2 = boxes_b[:, 3]

    # Координаты пересечения
    inter_x1 = np.maximum(a_x1, b_x1)
    inter_y1 = np.maximum(a_y1, b_y1)
    inter_x2 = np.minimum(a_x2, b_x2)
    inter_y2 = np.minimum(a_y2, b_y2)

    # Площадь пересечения
    inter_width = np.maximum(0, inter_x2 - inter_x1)
    inter_height = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Площади боксов
    area_a = (a_x2 - a_x1) * (a_y2 - a_y1)  # (N, 1)
    area_b = (b_x2 - b_x1) * (b_y2 - b_y1)  # (M,)

    # Площадь объединения
    union = area_a + area_b - inter_area

    # IoU
    iou_mat = inter_area / union

    # Заменяем NaN (где union=0) на 0
    iou_mat = np.nan_to_num(iou_mat, nan=0.0)

    return iou_mat