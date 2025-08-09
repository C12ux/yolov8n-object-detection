from deep_sort_realtime.deepsort_tracker import DeepSort

def get_tracker():
    return DeepSort(
        max_age=30,
        n_init=3,
        max_iou_distance=0.7,
        nn_budget=100
    )
