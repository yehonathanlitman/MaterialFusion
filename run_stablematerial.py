import os
import torch
import torchvision.transforms as T
import imageio.v2 as imageio
import numpy as np

from render import util

from stablematerial import StableMaterialPipeline, StableMaterialPipelineMV
from dataset import DatasetNERF

def predict_sv(pipe, args, cfg_scale=3):
    image_transforms = T.Compose(
        [
            T.ToTensor(),
            T.Resize((args.resolution, args.resolution), antialias = True),
            T.Normalize([0.5], [0.5])
        ]
    )
    res_dir = os.path.join(args.output_dir, "single_view")
    os.makedirs(res_dir, exist_ok=True)
    with torch.no_grad():
        #TODO: make a dataloader for this
        for f in list(set(glob.glob(os.path.join(args.data_path, "*.png"))
                          + glob.glob(os.path.join(args.data_path, "*.jpg")))
                          - set(glob.glob(os.path.join(args.data_path, "*_albedo*")))
                          - set(glob.glob(os.path.join(args.data_path, "*_normal*")))
                          - set(glob.glob(os.path.join(args.data_path, "*_orm*")))):
            mean_albedo, mean_orm = [], []
            img = imageio.imread(f)
            if img.shape[-1] == 4:
                img, mask = img[..., :3], T.ToTensor()(img[..., 3]).unsqueeze(0)
                mask = T.Resize((args.resolution, args.resolution), antialias=True)(mask)
            elif img.shape[-1] == 3:
                img, mask = img[..., :3], torch.ones((1, 1, args.resolution, args.resolution))
            else:
                raise Exception("Input image channel dimensions must be 3 or 4!")
            img = image_transforms(img).unsqueeze(0)
            img *= mask

            base, fname = os.path.split(f)
            temp_res_dir = os.path.join(res_dir, os.path.split(base)[1], fname.split(".")[0])
            os.makedirs(temp_res_dir, exist_ok=True)
            for pred_idx in range(args.num_predictions):
                pred_images = pipe.__call__(prompt_imgs=img, height=args.resolution, width=args.resolution, num_inference_steps=50, guidance_scale=cfg_scale, output_type="none").images
                if args.save_all_predictions:
                    for (albedo, orm) in zip(pred_images[0], pred_images[1]):
                        imageio.imwrite(os.path.join(temp_res_dir, f"{pred_idx}_albedo.png"), (albedo * 255).astype(np.uint8))
                        imageio.imwrite(os.path.join(temp_res_dir, f"{pred_idx}_orm.png"), (orm * 255).astype(np.uint8))
                pred_albedo = torch.tensor(pred_images[0]).permute(0,3,1,2)*mask
                pred_orm = torch.tensor(pred_images[1]).permute(0,3,1,2)*mask
                mean_albedo.append(pred_albedo)
                mean_orm.append(pred_orm)
            pred_albedo = torch.mean(torch.cat(mean_albedo), dim=0)
            pred_orm = torch.mean(torch.cat(mean_orm), dim=0)
            save_image(pred_orm, os.path.join(temp_res_dir, f"mean_orm_sv.png"))
            save_image(pred_albedo, os.path.join(temp_res_dir, f"mean_albedo_sv.png"))

def predict_mv(pipe, args, cfg_scale=3):
    image_transforms = T.Compose(
        [
            T.Resize((args.resolution, args.resolution), antialias = True),
            T.Normalize([0.5], [0.5])
        ]
    )
    test_dataset = DatasetNERF(os.path.join(args.data_path, "transforms_train.json"), args)
    if len(test_dataset) % args.num_views != 0:
        raise Exception("# of images in dataset must be divisible by the number of views")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=test_dataset.collate,
        batch_size=args.num_views,
    )
    obj_name = os.path.split(args.data_path)[1]
    res_dir = os.path.join(args.output_dir, "multi_view", obj_name)
    os.makedirs(res_dir, exist_ok=True)
    with torch.no_grad():
        for it, batch in enumerate(test_dataloader): #TODO: use tqdm here
            if batch["img"].shape[-1] == 4:
                imgs, masks = batch["img"][..., :3], batch["img"][..., 3].unsqueeze(1)
                masks = T.Resize((args.resolution, args.resolution), antialias=True)(masks)
            elif batch["img"].shape[-1] == 3:
                imgs, masks = batch["img"][..., :3], torch.ones((args.num_views, 1, args.resolution, args.resolution))
            else:
                raise Exception("Input image channel dimensions must be 3 or 4!")
            imgs = image_transforms(imgs.permute(0, 3, 1, 2)*masks)
            mean_albedo, mean_orm = [], []
            temp_res_dir = os.path.join(res_dir, f"{it*args.num_views}-{(it+1)*args.num_views - 1}")
            os.makedirs(temp_res_dir, exist_ok=True)
            for pred_idx in range(args.num_predictions):
                pred_images = pipe.__call__(prompt_imgs=imgs, poses=batch["T"], height=args.resolution, width=args.resolution, num_inference_steps=50, guidance_scale=cfg_scale, output_type="none").images
                if args.save_all_predictions:
                    for img_idx, (albedo, orm) in enumerate(zip(pred_images[0], pred_images[1])):
                        imageio.imwrite(os.path.join(temp_res_dir, f"{img_idx:03d}-{pred_idx}_albedo.png"), (albedo * 255).astype(np.uint8))
                        imageio.imwrite(os.path.join(temp_res_dir, f"{img_idx:03d}-{pred_idx}_orm.png"), (orm * 255).astype(np.uint8))
                pred_albedo = torch.tensor(pred_images[0]).permute(0,3,1,2)*masks
                pred_orm = torch.tensor(pred_images[1]).permute(0,3,1,2)*masks
                mean_albedo.append(pred_albedo)
                mean_orm.append(pred_orm)
            pred_albedo = torch.mean(torch.stack(mean_albedo), dim=0)
            pred_orm = torch.mean(torch.stack(mean_orm), dim=0)
            save_image(pred_orm, os.path.join(temp_res_dir, f"mean_orm_mv.png"))
            save_image(pred_albedo, os.path.join(temp_res_dir, f"mean_albedo_mv.png"))

if __name__ == "__main__":
    from torchvision.utils import save_image
    import glob
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory where predictions will be saved",
        default="out/stablematerial_pred"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        help="Path to Stable Material model directory",
        default="thebluser/stable-material"
    )
    parser.add_argument(
        "--mv_model_id",
        type=str,
        help="Path to Stable Material multi-view model directory",
        default="thebluser/stable-material-mv"
    )
    parser.add_argument(
        "--num_views",
        type=int,
        help="Number of views for multi-view attention (if >1 then MV model is used)",
        default=1
    )
    parser.add_argument(
        "--num_predictions",
        type=int,
        help="Number of times to predict the albedo/ORM before averaging",
        default=10
    )
    parser.add_argument(
        "--resolution", 
        type=int, 
        default=512
    )
    parser.add_argument(
        "--cam_near", 
        type=float, 
        default=0.1
    )
    parser.add_argument(
        "--cam_far", 
        type=float, 
        default=1000.0
    )
    parser.add_argument(
        "--save_all_predictions", 
        action='store_true',
        help="If used, the intermediate predictions used to compute the mean albedo/ORM will also be saved"
    )
    args = parser.parse_args()
    args.pre_load = True
    args.spp = 16
    args.cam_near_far = [args.cam_near, args.cam_far]

    if args.num_views > 1:
        pipe = StableMaterialPipelineMV.from_pretrained(args.mv_model_id, torch_dtype=torch.float16, trust_remote_code=True).to("cuda")
        predict_mv(pipe, args)
    else:
        pipe = StableMaterialPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16).to("cuda")
        predict_sv(pipe, args)

    print(f"Predictions are at {args.output_dir}")