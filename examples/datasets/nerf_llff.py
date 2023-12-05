""" 
"""

import numpy as np
import os, imageio
import torch
import torch.nn.functional as F
from .lmc import LMC
from .utils import Rays, Image

########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
        
        
def _load_data(basedir, subject_id, factor=None, width=None, height=None, load_imgs=True):

    data_dir = os.path.join(basedir, subject_id)
    
    poses_arr = np.load(os.path.join(data_dir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(data_dir, 'images', f) for f in sorted(os.listdir(os.path.join(data_dir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(data_dir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(data_dir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(data_dir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(data_dir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f)#, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3] for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs


def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
    


def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds
    

def load_llff_data(basedir, subject_id, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    poses, bds, imgs = _load_data(basedir, subject_id, factor=factor) # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc
    
    if recenter:
        poses = recenter_poses(poses)
        
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
        
    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)
    print(images.shape)

    hwf = poses[0,:3,-1]
    poses = poses[:,:3,:4]

    H, W, focal = hwf
    H, W = int(H), int(W)

    K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    all_indices = np.arange(images.shape[0])
    split_indices = {
        "test": all_indices[all_indices % 8 == 0],
        "train": all_indices[all_indices % 8 != 0],
    }

    return images, poses, K, split_indices


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    # shift the ray origins to the near plane by computing the intersection of each ray with the near plane
    # This is done by solving the equation `near + t * rays_d[...,2] = -rays_o[...,2]` for `t`,
    # where `t` is the distance from the ray origin to the near plane.
    # The ray origins are then shifted by `t * rays_d` to lie on the near plane.
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = (rays_o + t[..., None] * rays_d).detach()
    
    # Project to 2D
    o0 = rays_o[...,0] / rays_o[...,2]
    o1 = rays_o[...,1] / rays_o[...,2]
    o2 = near /  rays_o[...,2]
    # map to [-1, 1] with -1 / (W / (2 * focal))
    o0 = -1./(W/(2.*focal)) * o0
    o1 = -1./(H/(2.*focal)) * o1
    o2 = 1. + 2. * o2

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)
    
    return rays_o, rays_d

class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "test"]
    SUBJECT_IDS = [
        "fern",
        "flower",
        "fortress",
        "horns",
        "leaves",
        "orchids",
        "room_llff",
        "trex",
    ]

    OPENGL_CAMERA = True

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "gray",
        num_rays: int = None,
        near: float = 1.0,
        far: float = 6.0,
        batch_over_images: bool = True,
        factor: int = 4,
        sampling_type: str = "uniform",
        device: str = "cpu",
        minpct=0.1,
        lossminpc=0.1,
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random", "gray"]
        self.split = split
        self.num_rays = num_rays
        self.near = near
        self.far = far
        self.training = (num_rays is not None) and (
            split in ["train", "trainval"]
        )
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        
        self.images, self.camtoworlds, self.K, split_indices = load_llff_data(
            root_fp, subject_id, factor=factor
        )
        
        indices = split_indices[split]
        self.images = self.images[indices]
        self.camtoworlds = self.camtoworlds[indices]
        # to tensor
        self.images = torch.from_numpy(self.images).to(torch.uint8).to(device, non_blocking=True)
        self.sampling_type = sampling_type
        if self.training:
            bs = np.ceil(self.num_rays / self.images.shape[0]).astype(np.int32) 
            self.num_rays = bs * self.images.shape[0]

            if self.num_rays != num_rays:
                print("Warning: num_rays is not divisible by number of images. Setting num_rays to %d" % self.num_rays)
            
            self.const_img_id = torch.arange(0, self.images.shape[0], device=device).repeat_interleave(bs)
            if sampling_type == "lmc":
                self.lmc = LMC(images=self.images, num_rays=self.num_rays, const_img_id=self.const_img_id, 
                            device=device, minpct=minpct, lossminpc=lossminpc)
        self.simages = Image(self.images, device=device)
        self.camtoworlds = (
            torch.from_numpy(self.camtoworlds).to(torch.float32).to(device, non_blocking=True)
        )
        self.K = torch.tensor(self.K).to(torch.float32).to(device, non_blocking=True)
        self.height, self.width = self.images.shape[1:3]
        self.HEIGHT = self.height
        self.WIDTH = self.width
        self.device = device

    def __len__(self):
        return len(self.images)

    
    def update_num_rays(self, num_rays):
        self.num_rays = num_rays
    
    def get_origin_direction(self, image_id, y, x):
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [num_rays, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        # directions.requires_grad = True
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )
        return origins, viewdirs

    def __getitem__(self, index, net_grad=None, loss_per_pix=None):
        if self.sampling_type == "uniform":
            data = self.fetch_data(index)
        elif self.sampling_type == "lmc":
            data = self.fetch_data_lmc(net_grad=net_grad, loss_per_pix=loss_per_pix)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        pixels, rays = data["rgb"], data["rays"]

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=self.images.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.images.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.images.device)
            elif self.color_bkgd_aug == "gray":
                color_bkgd = torch.ones(3, device=self.images.device) * 0.5
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.images.device)

        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgb", "rays"]},
        }

    def compute_imgind_xy(self, index, num_rays):
        if self.training:
            if self.batch_over_images:
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(num_rays,),
                    device=self.images.device,
                )
            else:
                image_id = [index] * num_rays
            x = torch.randint(
                0, self.width, size=(num_rays,), device=self.images.device
            )
            y = torch.randint(
                0, self.height, size=(num_rays,), device=self.images.device
            )
            x = x.float()
            y = y.float()
            
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.width, device=self.images.device),
                torch.arange(self.height, device=self.images.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()
            x = x.float()
            y = y.float()


        return image_id, x, y


    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays

        image_id, x, y = self.compute_imgind_xy(index=index, num_rays=num_rays)

        # generate rays
        rgb = self.simages(image_id, y, x)  / 255.0
        origins, viewdirs = self.get_origin_direction(image_id, y, x)

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgb = torch.reshape(rgb, (num_rays, 3))
        else:
            origins = torch.reshape(origins, (self.height, self.width, 3))
            viewdirs = torch.reshape(viewdirs, (self.height, self.width, 3))
            rgb = torch.reshape(rgb, (self.height, self.width, 3))

        origins, viewdirs = ndc_rays(self.height, self.width, self.K[0][0], 1., origins, viewdirs)
        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgb": rgb,  # [h, w, 3] or [num_rays, 3]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "x": x,
            "y": y,
            "image_id": image_id
        }

    def fetch_data_lmc(self, net_grad=None, loss_per_pix=None):
        points_2d = self.lmc(net_grad, loss_per_pix)

        rgb = self.simages(self.const_img_id, points_2d[:, 0], points_2d[:, 1])  / 255.0
        points_2d.requires_grad = True
        x, y = points_2d[:, 1], points_2d[:, 0]
        
        origins, viewdirs = self.get_origin_direction(self.const_img_id, y, x)

        origins = torch.reshape(origins, (self.num_rays, 3))
        viewdirs = torch.reshape(viewdirs, (self.num_rays, 3))
        rgb = torch.reshape(rgb, (self.num_rays, 3))
        
        origins, viewdirs = ndc_rays(self.height, self.width, self.K[0][0], 1., origins, viewdirs)

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgb": rgb,  # [h, w, 3] or [num_rays, 3]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "x": x,
            "y": y,
            "image_id": self.const_img_id,
            "points_2d": points_2d
        }