from torch import nn
import torch
import superflex.functional as F
class StackedPVConv(nn.Module):
    def __init__(self, ctx):
        super(StackedPVConv, self).__init__()
        self.stacked_pvconv = torch.nn.Sequential(
            PVConv(ctx.l1),
            PVConv(ctx.l2),
            PVConv(ctx.l3)
            )
    
    def forward(self, coords):
        '''
        Args: 
            inputs: tuple of features and coords 
                coords:   B,3, num-points 
        Returns:
            fused_features: in (B,out-feat-dim,num-points)
        '''
        coords_transpose = coords.transpose(-1,-2)
        outputs_transpose, _ = self.stacked_pvconv((coords_transpose, coords_transpose))
        return outputs_transpose.transpose(-1,-2)

def swish(input):
    return input * torch.sigmoid(input)

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return swish(input)

class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, dim=1):
        super().__init__()
        if dim==1:
            conv = nn.Conv1d
        else:
            conv = nn.Conv2d
        bn = nn.GroupNorm 
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        layers = []
        for oc in out_channels:
            layers.append( conv(in_channels, oc, 1)) 
            layers.append(bn(8, oc))
            layers.append(Swish()) 
            in_channels = oc
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return (self.layers(inputs[0]), *inputs[1:])
        else:
            return self.layers(inputs)


class Voxelization(nn.Module):
    def __init__(self, resolution, normalize=True, eps=0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords):
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)
        if self.normalize:
            norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
        else:
            norm_coords = (norm_coords + 1) / 2.0
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        vox_coords = torch.round(norm_coords).to(torch.int32)
        return F.avg_voxelize(features, vox_coords, self.r), norm_coords

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')
    
class PVConv(nn.Module):

    def __init__(self, ctx):
        super(PVConv, self).__init__()
        
        self.in_channels = ctx.in_channels
        self.out_channels = ctx.out_channels
        self.kernel_size = ctx.kernel_size
        self.resolution = ctx.resolution
        
        self.voxelization = Voxelization(self.resolution, normalize=ctx.voxelization.normalize, eps=ctx.voxelization.eps)
        
        voxel_layers = [
            nn.Conv3d(self.in_channels, self.out_channels, self.kernel_size, stride=1, padding=self.kernel_size // 2),
            nn.GroupNorm(8, self.out_channels),
            Swish(),
            nn.Conv3d(self.out_channels, self.out_channels, self.kernel_size, stride=1, padding=self.kernel_size // 2),
            nn.GroupNorm(8, self.out_channels)
         ]
        
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_features = SharedMLP(self.in_channels, self.out_channels)

    def forward(self, inputs):  
        '''
        Args: 
            inputs: tuple of features and coords 
                features: B,feat-dim,num-points 
                coords:   B,3, num-points 
        Returns:
            fused_features: in (B, out-feat-dim, num-points)
            coords        : in (B, 3, num_points); same as the input coords
        '''
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = voxel_features + self.point_features(features)
        return (fused_features, coords)