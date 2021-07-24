import numpy as np

from detector.utils.box import box_iou


class ConfusionMatrix:
    """Confusion matrix based on bounding boxes.

    Example:

    .. code-block:: python

        import numpy as np
        ground_truth = np.array([
            # x_min, y_min, x_max, y_max, class_id
            [10, 10, 110, 110, 1],
            [50, 50, 150, 150, 2],
        ])
        predicted = np.array([
            # x_min, y_min, x_max, y_max, class_id, confidence
            [20, 20,  90,  90, 1, 0.6],
            [10, 10, 130, 130, 1, 0.55],
            [40, 40, 160, 160, 2, 0.3],
            [60, 60, 140, 140, 2, 0.8],
        ])
        metric_fn = ConfusionMatrix(num_classes=2)
        for _ in range(10):
            metric_fn.add(predicted, ground_truth)
        print(metric_fn.value())
        # Output:
        #   [[ 0.  0.  0.]
        #    [ 0. 10. 10.]
        #    [ 0.  0. 10.]]
    """

    def __init__(self, num_classes, iou_threshold=0.5, confidence_threshold=0.5):
        """
        Args:
            num_classes (float): number of classes in data
            iou_threshold (float): IOU threshold to use for filtering bboxes as matched.
            confidence_threshold (float): probability threshold to use for filtering class confidence.
        """
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        """Reset stored data."""
        self.confusion_matrix = np.zeros((self.num_classes + 1, self.num_classes + 1), dtype=np.int32)

    def add(self, predictions, ground_truth):
        """Add sample to evaluation.

        Args:
            predictions (np.ndarray): predictions for an image.
                Expected shapes - [N, 6] where N is a anchors.
                Each row should have structure [x_min, y_min, x_max, y_max, class_id, confidence].
            ground_truth (np.ndarray): actual bounding boxes,
                Expected shapes - [M, 5] where M is a number of bboxes on image.
                Each row should have structure [x_min, y_min, x_max, y_max, class_id]
        """
        predictions = predictions[predictions[:, 5] > self.confidence_threshold]
        gt_classes = ground_truth[:, 4].astype(np.int16)
        detection_classes = predictions[:, 4].astype(np.int16)

        all_ious = box_iou(ground_truth[:, :4], predictions[:, :4])
        want_idx = np.where(all_ious > self.iou_threshold)

        all_matches = []
        for i in range(want_idx[0].shape[0]):
            all_matches.append([want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]])

        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        for i, label in enumerate(ground_truth):
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                gt_class = gt_classes[i]
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.confusion_matrix[(gt_class), detection_class] += 1
            else:
                gt_class = gt_classes[i]
                self.confusion_matrix[self.num_classes, (gt_class)] += 1

        for i, detection in enumerate(predictions):
            if all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0:
                detection_class = detection_classes[i]
                self.confusion_matrix[detection_class, self.num_classes] += 1

    def value(self):
        """Compute confusion matrix.

        Returns:
            np.ndarray with confusion matrix
        """
        return np.copy(self.confusion_matrix)

    def __str__(self):
        rows = []
        for i in range(self.num_classes + 1):
            rows.append(" ".join(map(str, self.confusion_matrix[i])))
        return "\n".join(rows)
