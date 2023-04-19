import torch

class RenderNetwork(torch.nn.Module):
    def __init__(
        self,
        input_size,
        dir_count
    ):
        super().__init__()
        self.input_size = 3*input_size + input_size*3 
        self.layers_main = torch.nn.Sequential(
              torch.nn.Linear(self.input_size, 256),
              torch.nn.ReLU(),
              torch.nn.Linear(256, 256),
              torch.nn.ReLU(),
              torch.nn.Linear(256, 256),
              torch.nn.ReLU(),
              torch.nn.Linear(256, 256),
              torch.nn.ReLU(),
              torch.nn.Linear(256, 256),
              torch.nn.ReLU(),
        )
        
        self.layers_main_2 = torch.nn.Sequential(
              torch.nn.Linear(256 + self.input_size, 256),
              torch.nn.ReLU(),
              torch.nn.Linear(256, 256),
              torch.nn.ReLU(),
              torch.nn.Linear(256, 256),
              torch.nn.ReLU(),
              torch.nn.Linear(256, 256),
              torch.nn.ReLU(),     
        )
        
        self.layers_sigma = torch.nn.Sequential(
            torch.nn.Linear(256 + self.input_size, 128), #dodane wejscie tutaj moze cos pomoze
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
        
        self.layers_rgb = torch.nn.Sequential(
            torch.nn.Linear(256 + self.input_size + dir_count, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3),
        )

    def forward(self, triplane_code, viewdir):
        triplane_code = torch.cat([triplane_code, viewdir[0]], dim=-1)
        x = self.layers_main(triplane_code)
        x1 = torch.concat([x, triplane_code], dim=1)
        
        x = self.layers_main_2(x1)
        xs = torch.concat([x, triplane_code], dim=1)
        
        sigma = self.layers_sigma(xs)
        x = torch.concat([x, triplane_code], dim=1)
        rgb = self.layers_rgb(x)
        return torch.concat([rgb, sigma], dim=1)

class ImagePlane(torch.nn.Module):

    def __init__(self, focal, poses, images, count, device='cuda'):
            super(ImagePlane, self).__init__()

            self.pose_matrices = []
            self.K_matrices = []
            self.images = []
            self.centroids = []

            self.focal = focal
            for i in range(min(count, poses.shape[0])):
                M = poses[i]
                M = torch.from_numpy(M)                
                M = M @ torch.Tensor([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).to(M.device)
                self.centroids.append(M[0:3, 3])
                M = torch.inverse(M)
                M = M[0:3]
                self.pose_matrices.append(M) 

                image = images[i]
                image = torch.from_numpy(image)
                self.images.append(image.permute(2,0,1))
                self.size = float(image.shape[0])
                K = torch.Tensor([[self.focal.item(), 0, 0.5*image.shape[0]], [0, self.focal.item(), 0.5*image.shape[0]], [0, 0, 1]])

                self.K_matrices.append(K)

            self.pose_matrices = torch.stack(self.pose_matrices).to(device)
            self.K_matrices = torch.stack(self.K_matrices).to(device)
            self.image_plane = torch.stack(self.images).to(device)
            self.centroids = torch.stack(self.centroids).to(device) 


    def forward(self, points=None):
        if points.shape[0] == 1:
            points = points[0]

        points = torch.concat([points, torch.ones(points.shape[0], 1).to(points.device)], 1).to(points.device)
        ps = self.K_matrices @ self.pose_matrices @ points.T
        pixels = (ps/ps[:,None,2])[:,0:2,:]
        pixels = pixels / self.size
        pixels = torch.clamp(pixels, 0, 1)
        pixels = pixels * 2.0 - 1.0
        pixels = pixels.permute(0,2,1)

        feats = []
        for img in range(self.image_plane.shape[0]):
            feat = torch.nn.functional.grid_sample(
                self.image_plane[img].unsqueeze(0),
                pixels[img].unsqueeze(0).unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False)
            feats.append(feat)
        feats = torch.stack(feats).squeeze(1)
        pixels = pixels.permute(1,0,2)
        pixels = pixels.flatten(1)
        feats = feats.permute(2,3,0,1)
        feats = feats.flatten(2)
        
        cposes = self.centroids.flatten()
        feats = feats[0]
        
        feats = torch.cat((feats, cposes.unsqueeze(0).repeat(feats.shape[0], 1)), dim=1)

        return feats
    
class MultiImageNeRF(torch.nn.Module):
    
    def __init__(self, image_plane, count, dir_count):
        super(MultiImageNeRF, self).__init__()
        self.image_plane = image_plane
        self.render_network = RenderNetwork(count, dir_count)
        
        self.input_ch_views = dir_count
        
    def parameters(self):
        return self.render_network.parameters()

    def set_image_plane(self, ip):
        self.image_plane = ip
        
    def forward(self, x):
        input_pts, input_views = torch.split(x, [3, self.input_ch_views], dim=-1)
        x = self.image_plane(input_pts)
        return self.render_network(x, input_views)
    
    
