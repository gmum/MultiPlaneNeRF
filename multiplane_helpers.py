import torch
from run_nerf_helpers import *


class RenderNetwork(torch.nn.Module):
    def __init__(
            self,
            input_size,
            dir_count
    ):
        super().__init__()
        self.input_size = 3 * input_size + 3 * input_size + input_size * 2 + 25
        print("INPUT SIZE ", self.input_size)
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

        self.layers_main_3 = torch.nn.Sequential(
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
            torch.nn.Linear(256 + self.input_size, 128),  # dodane wejscie tutaj moze cos pomoze
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

        self.layers_rgb = torch.nn.Sequential(
            torch.nn.Linear(256 + self.input_size + dir_count + 25, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3),
        )

    def forward(self, triplane_code, dirs, ts):
        triplane_code = torch.concat([triplane_code], dim=1)
        x = self.layers_main(triplane_code)
        x1 = torch.concat([x, triplane_code], dim=1)

        x = self.layers_main_2(x1)
        xs = torch.concat([x, triplane_code], dim=1)

        x = self.layers_main_3(xs)
        xs = torch.concat([x, triplane_code], dim=1)

        sigma = self.layers_sigma(xs)
        x = torch.concat([x, triplane_code, dirs, ts], dim=1)
        rgb = self.layers_rgb(x)
        return torch.concat([rgb, sigma], dim=1)


class RenderNetworkEmbedded(torch.nn.Module):
    def __init__(
            self,
            input_size=100 * 3,
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

        self.layers_main2 = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
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
        x = self.layers_main2(x)
        sigma = self.layers_sigma(x)
        x = torch.concat([x, triplane_code], dim=1)
        rgb = self.layers_rgb(x)
        return torch.concat([rgb, sigma], dim=1)


class ImagePlanes(torch.nn.Module):

    def __init__(self, focal, poses, images, times, count, device='cuda'):
        super(ImagePlanes, self).__init__()

        self.count = count
        self.pose_matrices = []
        self.K_matrices = []
        self.images = []
        self.time_channels = []  # time channels

        self.focal = focal
        for i in range(min(count, poses.shape[0])):
            M = poses[i]
            M = torch.from_numpy(M)
            M = M @ torch.Tensor([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).to(M.device)
            M = torch.inverse(M)
            M = M[0:3]
            self.pose_matrices.append(M)

            # image = images[i][:, :, :-1]
            image = images[i]
            last_channel = images[i][:, :, -1:]
            last_channel_expanded = np.repeat(last_channel, 2, axis=2)
            image = np.concatenate((image, last_channel_expanded), axis=2)
            image = torch.from_numpy(image)
            self.images.append(image.permute(2, 0, 1))
            self.size = float(image.shape[0])
            K = torch.Tensor(
                [[self.focal.item(), 0, 0.5 * image.shape[0]], [0, self.focal.item(), 0.5 * image.shape[0]], [0, 0, 1]])

            self.K_matrices.append(K)

            time_channel = times[i]
            # time_channel = np.full(poses.shape[0], times[i], dtype=np.float32)
            # time_channel = images[i][:, :, -1:]

            # time_channel = times[i]  # Time channel, the last column
            # embedtime_fn, input_ch_time = get_embedder(10, 1)  # get embedder, arguments from run_nerf, create_mi_nerf, except of last argument
            # time_channel = torch.from_numpy(time_channel).to(device)
            # embed_time_channel = embedtime_fn(time_channel)
            # embed_time_channel_mean = torch.mean(embed_time_channel, dim=2, keepdim=True)
            # self.time_channels.append(embed_time_channel_mean.permute(2, 0, 1))

            time_channel = torch.tensor(time_channel)
            self.time_channels.append(time_channel)

        self.pose_matrices = torch.stack(self.pose_matrices).to(device)
        self.K_matrices = torch.stack(self.K_matrices).to(device)
        self.image_plane = torch.stack(self.images).to(device)
        self.time_channels = torch.stack(self.time_channels).to(device)  # list to tensor
        self.time_channels = torch.reshape(self.time_channels, (count, 1))
        print(self.time_channels.shape)

    def forward(self, points=None, ts=None):
        if points.shape[0] == 1:
            points = points[0]

        ''' ts to jest konkretna chwila czasowa '''
        points = torch.concat([points, torch.ones(points.shape[0], 1).to(points.device)], 1).to(points.device)
        ps = self.K_matrices @ self.pose_matrices @ points.T
        pixels = (ps / ps[:, None, 2])[:, 0:2, :]
        pixels = pixels / self.size
        pixels = torch.clamp(pixels, 0, 1)
        pixels = pixels * 2.0 - 1.0
        pixels = pixels.permute(0, 2, 1)

        ts_time = ts[0][0].item()

        feats = []
        for img in range(min(self.count, self.image_plane.shape[0])):
            time1 = (self.time_channels[img] - ts_time).abs().item()
            time2 = time1
            if img != self.count - 1:
                time2 = (self.time_channels[img + 1] - ts_time).abs().item()

            # Interpolacja pomiędzy dwiema klatkami czasowymi
            weight1 = 1.0 - time1
            weight2 = 1.0 - time2

            # Dwa obrazy odpowiadające dwóm klatkom czasowym
            frame1 = self.image_plane[img]
            frame2 = frame1
            if img != self.count - 1:
                frame2 = self.image_plane[img + 1]

            # Interpolacja między dwiema klatkami
            interpolated_frame = weight1 * frame1 + weight2 * frame2
            if weight1 > 0 or weight2 > 0:
                min_val = interpolated_frame.min()
                max_val = interpolated_frame.max()
                interpolated_frame = (interpolated_frame - min_val) / (max_val - min_val)

            # Przetwarzanie
            feat = F.grid_sample(
                interpolated_frame.unsqueeze(0),
                pixels[img].unsqueeze(0).unsqueeze(0),
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )
            feats.append(feat)

        feats = torch.stack(feats).squeeze(1)

        pixels = pixels.permute(1, 0, 2)
        pixels = pixels.flatten(1)

        feats = feats.permute(2, 3, 0, 1)
        feats = feats.flatten(2)

        feats = torch.cat((feats[0], pixels, ts), 1)

        # time = ts[0].item()
        # time_id = torch.where(self.time_channels == time)[0].item()
        # pixels_at_exact_time = pixels[time_id]
        #
        # feat = torch.nn.functional.grid_sample(
        #     self.image_plane[time_id].unsqueeze(0),
        #     pixels_at_exact_time.unsqueeze(0).unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False)
        # feats.append(feat)
        #
        # feats = torch.stack(feats).squeeze(1)
        #
        # feats = feats.permute(2, 3, 0, 1)
        # feats = feats.flatten(2)
        #
        # feats = torch.cat((feats[0], pixels_at_exact_time, ts), 1)

        return feats


class LLFFImagePlanes(torch.nn.Module):

    def __init__(self, hwf, poses, images, count, device='cuda'):
        super(LLFFImagePlanes, self).__init__()

        self.pose_matrices = []
        self.K_matrices = []
        self.images = []

        self.H, self.W, self.focal = hwf

        for i in range(min(count, poses.shape[0])):
            M = poses[i]
            M = torch.from_numpy(M)
            M = torch.cat([M, torch.Tensor([[0, 0, 0, 1]]).to(M.device)], dim=0)

            M = M @ torch.Tensor([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).to(M.device)
            M = torch.inverse(M)
            M = M[0:3]
            self.pose_matrices.append(M)

            image = images[i]
            image = torch.from_numpy(image)
            self.images.append(image.permute(2, 0, 1))
            self.size = float(image.shape[0])
            K = torch.Tensor([[self.focal, 0, 0.5 * self.W], [0, self.focal, 0.5 * self.H], [0, 0, 1]])

            self.K_matrices.append(K)

        self.pose_matrices = torch.stack(self.pose_matrices).to(device)
        self.K_matrices = torch.stack(self.K_matrices).to(device)
        self.image_plane = torch.stack(self.images).to(device)

    def forward(self, points=None):

        if points.shape[0] == 1:
            points = points[0]

        points = torch.concat([points, torch.ones(points.shape[0], 1).to(points.device)], 1).to(points.device)
        ps = self.K_matrices @ self.pose_matrices @ points.T
        pixels = (ps / ps[:, None, 2])[:, 0:2, :]
        pixels[:, 0] = torch.div(pixels[:, 0], self.W)
        pixels[:, 1] = torch.div(pixels[:, 1], self.H)
        pixels = torch.clamp(pixels, 0, 1)
        pixels = pixels * 2.0 - 1.0
        pixels = pixels.permute(0, 2, 1)

        feats = []
        for img in range(self.image_plane.shape[0]):
            feat = torch.nn.functional.grid_sample(
                self.image_plane[img].unsqueeze(0),
                pixels[img].unsqueeze(0).unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False)
            feats.append(feat)
        feats = torch.stack(feats).squeeze(1)
        pixels = pixels.permute(1, 0, 2)
        pixels = pixels.flatten(1)
        feats = feats.permute(2, 3, 0, 1)
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

    def forward(self, x, ts):
        # print("Calling now", x.shape)
        input_pts, input_views = torch.split(x, [3, self.input_ch_views], dim=-1)
        # input_pts_with_time = torch.cat([input_pts, ts[0]], dim=1)
        ts_more_channels = ts[0].expand(-1, 25)
        x = self.image_plane(input_pts, ts_more_channels)
        return self.render_network(x, input_views, ts_more_channels), torch.zeros_like(input_pts[:, :3])


class EmbeddedMultiImageNeRF(torch.nn.Module):

    def __init__(self, image_plane, count):
        super(EmbeddedMultiImageNeRF, self).__init__()
        self.image_plane = image_plane
        self.render_network = RenderNetworkEmbedded(count * 3)

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


