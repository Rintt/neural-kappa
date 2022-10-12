import torch
import torch.nn.functional as F

def casteljau(control_points: torch.Tensor, t):
    points = control_points.clone()
    n = control_points.shape[0]
    for k in range(1, n):
        for i in range(n - k):
            points[i] = (1.0-t)*points[i] + t*points[i+1]
        if k == n - 2:
            tangent = points[1] - points[0]
    return points[0], tangent

def casteljau_diff(control_points: torch.Tensor, t):
    # FIXME: Differentiation w.r.t. t is super easy: just use the tangent
    num_channels = control_points.shape[1]
    filters = torch.stack([1-t, t], dim=-1).unsqueeze(0).repeat((num_channels, 1, 1))
    points = control_points.permute(1, 0).unsqueeze(0)
    n = control_points.shape[0]
    for k in range(1, n):
        points = F.conv1d(points, filters, groups=num_channels)
        if k == n - 2:
            tangent = points[0, :, 1] - points[0, :, 0]
    return points[0, :, 0], tangent