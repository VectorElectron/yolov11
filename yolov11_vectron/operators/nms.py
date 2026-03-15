import torch
import torch.nn as nn

def conf_filter(y_out, conf_threshold=0.25, top_k=1024):
    B, C, N = y_out.shape
    y_out = y_out.transpose(1, 2)
    y_box = y_out[:, :, :4]
    y_cls = y_out[:, :, 4:]
    
    scores, class_ids = torch.max(y_cls, dim=-1)
    batch_ids = torch.arange(B, device=y_out.device).view(-1, 1).expand(-1, N)

    f_boxes = y_box.reshape(-1, 4)
    f_scores = scores.reshape(-1)
    f_clss = class_ids.reshape(-1)
    f_batch = batch_ids.reshape(-1)

    mask = f_scores > conf_threshold
    f_boxes = f_boxes[mask]
    f_scores = f_scores[mask]
    f_clss = f_clss[mask]
    f_batch = f_batch[mask]

    # --- 插入虚假目标 (Dummy Object) ---
    d_box = torch.zeros((1, 4), device=y_out.device)
    d_score = torch.tensor([-1.0], device=y_out.device)
    d_cls = torch.zeros((1,), device=y_out.device, dtype=f_clss.dtype)
    d_batch = torch.zeros((1,), device=y_out.device, dtype=f_batch.dtype)

    f_boxes = torch.cat([f_boxes, d_box], dim=0)
    f_scores = torch.cat([f_scores, d_score], dim=0)
    f_clss = torch.cat([f_clss, d_cls], dim=0)
    f_batch = torch.cat([f_batch, d_batch], dim=0)

    num_candidates = f_scores.shape[0]
    k = torch.clamp(torch.tensor(top_k, device=y_out.device), max=num_candidates)
    
    f_scores, topk_indices = torch.topk(f_scores, k, sorted=False)
    f_boxes = f_boxes[topk_indices]
    f_clss = f_clss[topk_indices]
    f_batch = f_batch[topk_indices]

    half_wh = f_boxes[:, 2:4] / 2
    bboxes = torch.cat([f_boxes[:, :2] - half_wh, f_boxes[:, :2] + half_wh], dim=-1)
    
    return bboxes, f_scores, f_clss, f_batch

def iou_matrix(boxes):
    # 使用广播计算 (M, 1, 4) vs (1, M, 4)
    b1 = boxes.unsqueeze(1)
    b2 = boxes.unsqueeze(0)
    
    inter_min = torch.max(b1[:, :, :2], b2[:, :, :2])
    inter_max = torch.min(b1[:, :, 2:], b2[:, :, 2:])
    inter_wh = torch.clamp(inter_max - inter_min, min=0)
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]
    
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = area.unsqueeze(1) + area.unsqueeze(0) - inter_area
    return inter_area / (union_area + 1e-7)

def graph_lut(iou_mat, scores, clss, batch, iou_threshold=0.45):
    N = iou_mat.size(0)
    # 建立 Batch & Class 隔离掩码
    batch_match = (batch.unsqueeze(1) == batch.unsqueeze(0))
    class_match = (clss.unsqueeze(1) == clss.unsqueeze(0))
    iou_match = iou_mat > iou_threshold
    
    combined_mask = batch_match & class_match & iou_match
    score_mask = scores.unsqueeze(0) > scores.unsqueeze(1)
    
    potential_targets = combined_mask & score_mask
    
    # 寻找更高分目标，无目标处设为 -1
    weighted_scores = torch.where(potential_targets, scores.unsqueeze(0), torch.tensor(-1.0, device=scores.device))
    initial_lut = torch.argmax(weighted_scores, dim=1)
    
    no_target = torch.all(~potential_targets, dim=1)
    return torch.where(no_target, torch.arange(N, device=scores.device), initial_lut)

def nms(y_pred, scale, conf_threshold=0.25, iou_threshold=0.45, top_k=1024):
    bboxes, scores, clss, batch = conf_filter(y_pred, conf_threshold, top_k)
    M = bboxes.size(0)
    ioumat = iou_matrix(bboxes)
    lut = graph_lut(ioumat, scores, clss, batch, iou_threshold)
    for _ in range(4): lut = lut[lut]
    indices = torch.arange(M, device=bboxes.device)
    score_sums = torch.zeros(M, device=bboxes.device)
    score_sums.scatter_add_(0, lut, scores) 
    weighted_boxes = torch.zeros_like(bboxes)
    for i in range(4):
        weighted_boxes[:, i].scatter_add_(0, lut, bboxes[:, i] * scores)
    safe_score_sums = torch.clamp(score_sums, min=1e-7)
    avg_boxes = weighted_boxes / safe_score_sums.view(-1, 1)
    final_mask = (score_sums > 0) & (lut == indices)
    
    return avg_boxes[final_mask] * torch.cat([scale, scale]), \
        scores[final_mask], clss[final_mask].to(torch.int32), batch[final_mask].to(torch.int32)


class YOLO_PostProcess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, grid, scale, conf_threshold, iou_threshold, top_k):
        return nms(grid, scale, conf_threshold, iou_threshold, top_k)

if __name__ == '__main__':
    model = YOLO_PostProcess()

    # 1. 定义模拟输入
    y_pred_dummy = torch.randn(1, 84, 8400) * 0
    conf_thres_dummy = torch.tensor(0.25, dtype=torch.float32)
    iou_thres_dummy = torch.tensor(0.45, dtype=torch.float32)
    scale_dummy = torch.tensor([1,1], dtype=torch.float32)
    top_k_dummy = torch.tensor(1024, dtype=torch.int32)

    # 组合成输入元组
    dummy_inputs = (y_pred_dummy, scale_dummy, conf_thres_dummy, iou_thres_dummy, top_k_dummy)

    # 2. 导出 ONNX
    torch.onnx.export(
        model, 
        dummy_inputs, 
        "../model/yolo_nms.onnx",
        export_params=True,
        dynamo=False,
        opset_version=17,
        input_names=['grid', 'scale', 'conf_thr', 'iou_thr', 'top_k'], # 暴露为输入
        output_names=['boxes', 'scores', 'clss', 'batchs'],
        dynamic_axes={
            'grid': {0: 'batch', 1: 'features', 2: 'ngrid'},
            'boxes': {0: 'num_dets'},
            'scores': {0: 'num_dets'},
            'class': {0: 'num_dets'},
            'batchs': {0: 'num_dets'}
        }
    )

    print("后处理模型已成功导出")
