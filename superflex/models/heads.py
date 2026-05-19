import torch.nn as nn
import torch.nn.functional as F
import torch

class SuperFlexHead(nn.Module):
    """Head for Superquadrics Prediction"""
    
    def __init__(self, emb_dims, ctx):
        super(SuperFlexHead, self).__init__()
        self.emb_dims = emb_dims
        self.scale_head = nn.Linear(emb_dims, 3)
        self.shape_head = nn.Linear(emb_dims, 2)
        self.rot_head = nn.Linear(emb_dims, 4)
        self.t_head = nn.Linear(emb_dims, 3)
        self.exist_head = nn.Linear(emb_dims, 1)
        
        self.extended = getattr(ctx, 'extended', False)
        if self.extended:
            extended_non_zero_init = getattr(ctx, 'extended_non_zero_init', False)
            self.tapering_head = nn.Linear(emb_dims, 2)
            self.bending_k_head = nn.Linear(emb_dims, 3)
            self.bending_a_head = nn.Linear(emb_dims, 6)
            
            if extended_non_zero_init:            
                # Initialize tapering and bending heads with small non-zero values
                nn.init.normal_(self.tapering_head.weight, mean=0.0, std=1e-4)
                nn.init.normal_(self.tapering_head.bias, mean=0.0, std=1e-4)
                nn.init.normal_(self.bending_a_head.weight, mean=0.0, std=1e-3)
                nn.init.normal_(self.bending_a_head.bias, mean=0.0, std=1e-3)
                # Initialize the 'cos' components (indices 1, 3, 5) near 1.0 so length is non-zero
                nn.init.normal_(self.bending_a_head.bias.data[1::2], mean=1.0, std=1e-3)
                nn.init.normal_(self.bending_k_head.weight, mean=0.0, std=1e-3)
                nn.init.constant_(self.bending_k_head.bias, -3.0)
            else:
                # Initialize tapering and bending heads: tapering and bending_a to 0,
                # and initialize bending_k bias to -6 so sigmoid produces a small
                # initial bending value (sigmoid(-6) ~ 0.0025 -> near zero)
                nn.init.zeros_(self.tapering_head.weight)
                nn.init.zeros_(self.tapering_head.bias)
                nn.init.zeros_(self.bending_a_head.weight)
                nn.init.zeros_(self.bending_a_head.bias)
                # Initialize the 'cos' components (indices 1, 3, 5) to 1.0 so length is non-zero
                self.bending_a_head.bias.data[1::2] = 1.0
                nn.init.zeros_(self.bending_k_head.weight)
                nn.init.constant_(self.bending_k_head.bias, -6.0)


    def forward(self, x):
        scale_pre_activation = self.scale_head(x)
        scale = self.scale_activation(scale_pre_activation)
        
        shape_before_activation = self.shape_head(x)
        shape = self.shape_activation(shape_before_activation)

        q = F.normalize(self.rot_head(x), dim=-1, p=2)
        rotation = self.quat2mat(q)            

        translation = self.t_head(x)

        exist = self.exist_activation(self.exist_head(x))

        out_dict = {"scale": scale, "shape": shape, "rotate": rotation, "trans": translation, "exist": exist}
        if self.extended:
            tapering = self.tapering_activation(self.tapering_head(x))
            bending_k = self.bending_k_activation(self.bending_k_head(x), scale, tapering)
            bend_a_raw = self.bending_a_head(x)
            bend_a_raw = bend_a_raw.view(*bend_a_raw.shape[:-1], 3, 2)
            bending_a = self.bending_a_activation(bend_a_raw)
            out_dict.update({"tapering": tapering, "bending_k": bending_k, "bending_a": bending_a})
        return out_dict
        
    @staticmethod  
    def quat2mat(quat):
        """Normalize the quaternion and convert it to rotation matrix"""
        B = quat.shape[0]
        N = quat.shape[1]
        quat = F.normalize(quat, dim=2)
        quat = quat.contiguous().view(-1,4)
        w, x, y, z = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z
        rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                            2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                            2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, N, 3, 3)
        rotMat = rotMat.view(B,N,3,3)
        return rotMat
    
    @staticmethod 
    def scale_activation(x):
        return torch.sigmoid(x) 
    
    @staticmethod 
    def shape_activation(x):
        return 0.1 + 1.8 * torch.sigmoid(x)
    
    @staticmethod 
    def exist_activation(x):
        return torch.sigmoid(x)
    
    @staticmethod 
    def tapering_activation(x):
        return torch.tanh(x)
    
    @staticmethod 
    def bending_k_activation(x, scale, taper):
        s = scale#.detach()
        tap = taper#.detach()
        s_x, s_y, s_z = s[..., 0:1], s[..., 1:2], s[..., 2:3]
        
        # Apply tapering to effective scales
        s_x_eff = s_x * (1 + torch.abs(tap[..., 0:1]))
        s_y_eff = s_y * (1 + torch.abs(tap[..., 1:2]))
        s_z_eff = s_z
    
        # Thickness Constraint
        t_x = torch.amax(torch.cat([s_y_eff, s_z_eff], dim=-1), dim=-1, keepdim=True)
        t_y = torch.amax(torch.cat([s_x_eff, s_z_eff], dim=-1), dim=-1, keepdim=True)
        t_z = torch.amax(torch.cat([s_x_eff, s_y_eff], dim=-1), dim=-1, keepdim=True)
        thickness_limit = torch.cat([t_x, t_y, t_z], dim=-1)
        
        # Global Loop Constraint (Circumference > Length)
        loop_limit = torch.cat([s_x_eff, s_y_eff, s_z_eff], dim=-1) / torch.pi
        
        min_radius_xyz = torch.maximum(thickness_limit, loop_limit)
        
        # Reorder to (z, x, y) for bending channels
        min_radius = torch.cat([
            min_radius_xyz[..., 2:3],
            min_radius_xyz[..., 0:1],
            min_radius_xyz[..., 1:2],
        ], dim=-1)
        return torch.sigmoid(x) / min_radius
    
    @staticmethod 
    def bending_a_activation(x):
        # x is shape (..., 3, 2) representing (sin, cos)
        # atan2 natively returns an angle in the range [-pi, pi]
        return torch.atan2(x[..., 0], x[..., 1])