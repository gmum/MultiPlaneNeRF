import torch

class RenderNetwork(torch.nn.Module):
    def __init__(
        self,
        input_size,
        dir_count
    ):
        super().__init__()
        self.input_size = 3*input_size + input_size*2
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

    def forward(self, triplane_code, dirs):
        x = self.layers_main(triplane_code)
        x1 = torch.concat([x, triplane_code], dim=1)
        
        x = self.layers_main_2(x1)
        xs = torch.concat([x, triplane_code], dim=1)
        
        sigma = self.layers_sigma(xs)
        x = torch.concat([x, triplane_code, dirs], dim=1)
        rgb = self.layers_rgb(x)
        return torch.concat([rgb, sigma], dim=1)
    
    
class RenderNetworkEmbedded(torch.nn.Module):
    def __init__(
        self,
        input_size=100*3,
    ):
        input_size = input_size + 200 + 32
        super().__init__()
        self.layers_main = torch.nn.Sequential(
              torch.nn.Linear(input_size, 256),
              torch.nn.ReLU(),
              torch.nn.Linear(256, 256),
              torch.nn.ReLU(),
              torch.nn.Linear(256, 256),
              torch.nn.ReLU(),
              
            )
        self.layers_sigma = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
        self.layers_rgb = torch.nn.Sequential(
            torch.nn.Linear(256 + input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3),
        )

    def forward(self, triplane_code):
        x = self.layers_main(triplane_code)
        sigma = self.layers_sigma(x)
        x = torch.concat([x, triplane_code], dim=1)
        rgb = self.layers_rgb(x)
        return torch.concat([rgb, sigma], dim=1)

class ImagePlanes(torch.nn.Module):

    def __init__(self, focal, poses, images, count, device='cuda'):
        super(ImagePlanes, self).__init__()

        self.pose_matrices = []
        self.K_matrices = []
        self.images = []

        self.focal = focal
        for i in range(min(count, poses.shape[0])):
            M = poses[i]
            M = torch.from_numpy(M)                
            M = M @ torch.Tensor([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).to(M.device)
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
        feats = torch.cat((feats[0], pixels), 1)
        return feats
    
    
class LLFFImagePlanes(torch.nn.Module):

    def __init__(self, hwf, poses, images, count, device='cuda'):
        super(LLFFImagePlanes, self).__init__()

        self.pose_matrices = []
        self.K_matrices = []
        self.images = []

        self.H, self.W, self.focal  = hwf
        
        for i in range(min(count, poses.shape[0])):
            M = poses[i]
            M = torch.from_numpy(M)
            M = torch.cat([M, torch.Tensor([[0,0,0,1]]).to(M.device)], dim=0)
                
            M = M @ torch.Tensor([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).to(M.device)
            M = torch.inverse(M)
            M = M[0:3]
            self.pose_matrices.append(M) 

            image = images[i]
            image = torch.from_numpy(image)
            self.images.append(image.permute(2,0,1))
            self.size = float(image.shape[0])
            K = torch.Tensor([[self.focal, 0, 0.5*self.W], [0, self.focal, 0.5*self.H], [0, 0, 1]])

            self.K_matrices.append(K)

        self.pose_matrices = torch.stack(self.pose_matrices).to(device)
        self.K_matrices = torch.stack(self.K_matrices).to(device)
        self.image_plane = torch.stack(self.images).to(device)


    def forward(self, points=None):
        
        if points.shape[0] == 1:
            points = points[0]

        points = torch.concat([points, torch.ones(points.shape[0], 1).to(points.device)], 1).to(points.device)
        ps = self.K_matrices @ self.pose_matrices @ points.T
        pixels = (ps/ps[:,None,2])[:,0:2,:]
        pixels[:, 0] = torch.div(pixels[:, 0], self.W)
        pixels[:, 1] = torch.div(pixels[:, 1], self.H)
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
        feats = torch.cat((feats[0], pixels), 1)
        return feats
    
class ImageEmbedder(torch.nn.Module):
    def __init__(self):
        super(ImageEmbedder, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((4, 4)),
            torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Flatten(),
            torch.nn.Linear(625, 32)
        )
        
    def forward(self, input_image):
        input_image = torch.from_numpy(input_image).to('cuda')
        input_image = input_image.permute(2, 0, 1)
        return self.model(input_image)
    
class MultiImageNeRF(torch.nn.Module):
    
    def __init__(self, image_plane, count, dir_count):
        super(MultiImageNeRF, self).__init__()
        self.image_plane = image_plane
        self.render_network = RenderNetwork(count, dir_count)
        
        self.input_ch_views = dir_count
        
    def parameters(self):
        return self.render_network.parameters()
        
    def forward(self, x):
        input_pts, input_views = torch.split(x, [3, self.input_ch_views], dim=-1)
        x = self.image_plane(input_pts)
        return self.render_network(x, input_views)

class EmbeddedMultiImageNeRF(torch.nn.Module):
    
    def __init__(self, image_plane, count):
        super(EmbeddedMultiImageNeRF, self).__init__()
        self.image_plane = image_plane
        self.render_network = RenderNetworkEmbedded(count*3)
        
    def parameters(self):
        return self.render_network.parameters()
        
    def set_embedding(self, emb):
        self.embedding = emb
        
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        x = self.image_plane(input_pts)
        e = self.embedding.repeat(x.shape[0], 1)
        x = torch.cat([x, e], -1)
        return self.render_network(x, input_views)
    
    
