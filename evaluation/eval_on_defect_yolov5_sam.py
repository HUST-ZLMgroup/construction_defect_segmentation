import argparse
import os
import time
import json
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import contextlib
# YOLOv5
#from yolov5.models.common import DetectMultiBackend
#from yolov5.utils.general import non_max_suppression, scale_boxes
#from yolov5.utils.torch_utils import select_device
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "yolov5"))
from yolo_eval import YoloEvaluator
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# SAM
from segment_anything import build_sam, build_sam_enhanced, SamPredictor, sam_model_registry


# ---------------- COCO Dataset ---------------- #
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file):
        super().__init__(img_folder, ann_file)

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        file_name = self.coco.loadImgs(image_id)[0]["file_name"]
        return img, target, image_id, file_name

def visualize_and_save(image, boxes, scores, masks, labels, cat_id_to_name, save_path, ious=None):

    vis_img = image.copy()

    for idx, (box, score, mask, label) in enumerate(zip(boxes, scores, masks, labels)):

        x1, y1, x2, y2 = map(int, box.tolist())

        # ---------- bbox ----------
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ---------- label ----------
        class_name = cat_id_to_name.get(int(label), str(int(label)))
        #text = f"{int(label)} {class_name} {score:.2f}"
        text = f"score:{score:.2f}"

        cv2.putText(
            vis_img,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
        
        if ious is not None:
            iou_text = f"IoU:{ious[idx]:.2f}"
            (tw, th), _ = cv2.getTextSize(iou_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x1
            text_y = y2 + th + 5
            h_img, w_img = vis_img.shape[:2]
            if text_y > h_img:
                text_y = y2 - 5
                
            #cv2.putText( vis_img, iou_text, (x2 - 60, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(vis_img,iou_text,(text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 255),1)

        # ---------- mask ----------
        mask = mask[0]  # [1, H, W] -> [H, W]
        colored_mask = np.zeros_like(vis_img)
        colored_mask[:, :, 1] = mask * 255  

        vis_img = cv2.addWeighted(vis_img, 1.0, colored_mask, 0.4, 0)

    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, vis_img)
        
def visualize_yolo(image, boxes, scores, labels, cat_id_to_name, save_path):
    vis_img = image.copy()

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box.tolist())

        # ---------- bbox ----------
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ---------- label ----------
        class_name = cat_id_to_name.get(int(label), str(int(label)))
        #text = f"{int(label)} {class_name} {score:.2f}"
        text = f"{score:.2f}"

        cv2.putText(
            vis_img,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, vis_img)
    
def visualize_gt(image, gt_masks, gt_boxes, gt_labels, cat_id_to_name, save_path):
    vis_img = image.copy()
    H, W = vis_img.shape[:2]

    for mask, box, label in zip(gt_masks, gt_boxes, gt_labels):

        # resize mask
        if mask.shape != (H, W):
            mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

        # ----------  mask ----------
        colored_mask = np.zeros_like(vis_img)
        colored_mask[..., 0] = mask * 255  # RGB: 红

        vis_img = cv2.addWeighted(vis_img, 1.0, colored_mask, 0.4, 0)

        # ---------- bbox ----------
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # ---------- label ----------
            class_name = cat_id_to_name.get(label, str(label))
            #text = f"class {label} {class_name}"
            text = f"{class_name}"

            cv2.putText(
                vis_img,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, vis_img)
    
def compute_iou_matrix(pred_masks, gt_masks):
    ious = np.zeros((len(pred_masks), len(gt_masks)))

    for i, pm in enumerate(pred_masks):
        pm = pm[0].astype(bool)

        for j, gm in enumerate(gt_masks):
            gm = gm.astype(bool)

            if pm.shape != gm.shape:
                gm = cv2.resize(gm.astype(np.uint8), (pm.shape[1], pm.shape[0]), interpolation=cv2.INTER_NEAREST)

            inter = np.logical_and(pm, gm).sum()
            union = np.logical_or(pm, gm).sum()

            if union > 0:
                ious[i, j] = inter / union

    return ious
# ---------------- Main ---------------- #
def main(args):

    device = select_device(args.device)

    print(f"Image path: {args.image_dir}")
    print(f"Anno path: {args.anno_path}")
    # Load YOLO
    yolo = DetectMultiBackend(args.yolo_checkpoint, device=device)
    yolo.eval()

    # Load SAM
    model_type = "vit_h" 

    use_sam_enhanced = args.use_sam_enhanced
    if use_sam_enhanced:
        sam_enhanced_checkpoint = args.sam_enhanced_checkpoint
        predictor = SamPredictor(build_sam_enhanced(checkpoint=sam_enhanced_checkpoint).to(args.device))
    else:
        sam_checkpoint = args.sam_checkpoint
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(args.device))
    # Dataset
    dataset = CocoDetection(args.image_dir, args.anno_path)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    coco_gt = dataset.coco

    start = time.time()
    
    # Visualization Dir 
    yolo_name = os.path.basename(args.yolo_checkpoint).replace(".pt", "")
    
    if args.use_sam_enhanced:
        sam_name = os.path.basename(args.sam_enhanced_checkpoint).replace(".pth", "")
    else:
        sam_name = os.path.basename(args.sam_checkpoint).replace(".pth", "")

    dataset_name = os.path.normpath(args.image_dir).replace(os.sep, "_")
    exp_name = f"{yolo_name}+{sam_name}"
    output_root = os.path.join("seginw_output", exp_name, dataset_name)
    vis_dir = os.path.join(output_root, "visualization")
    #eval_dir = os.path.join(output_root, "evaluation")

    #os.makedirs(vis_dir, exist_ok=True)
    #os.makedirs(eval_dir, exist_ok=True)
    origin_dir = os.path.join(vis_dir, "origin")
    yolo_dir = os.path.join(vis_dir, "yolo")
    yolo_sam_dir = os.path.join(vis_dir, "yolo+sam")
    gt_dir = os.path.join(vis_dir, "gt")

    os.makedirs(origin_dir, exist_ok=True)
    os.makedirs(yolo_dir, exist_ok=True)
    os.makedirs(yolo_sam_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    print("Visualization dir:", vis_dir)
    #print("Evaluation dir:", eval_dir)
    
    #evaluator = YoloEvaluator(dataset.coco, iou_types=("bbox", "segm"), output_dir=eval_dir)
    evaluator = YoloEvaluator(dataset.coco, iou_types=("bbox", "segm"))

    for i, (img_pil, _, image_id, file_name) in enumerate(data_loader):
        # get GT mask
        ann_ids = coco_gt.getAnnIds(imgIds=image_id)
        anns = coco_gt.loadAnns(ann_ids)
        
        cat_id_to_name = {cat["id"]: cat["name"] for cat in dataset.coco.loadCats(dataset.coco.getCatIds())}
        
        # convert GT mask
        gt_masks = []
        gt_boxes = []
        gt_labels = []
        for ann in anns:
            #print(ann["category_id"]) test if single class id
            # ---------- mask ----------
            gt_mask = coco_gt.annToMask(ann)
            gt_masks.append(gt_mask)

            # ---------- bbox (COCO format: x, y, w, h) ----------
            x, y, w, h = ann["bbox"]
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            gt_boxes.append([x1, y1, x2, y2])

            # ---------- label ----------
            gt_labels.append(ann["category_id"])

        # Load raw image
        image_path = os.path.join(args.image_dir, file_name)
        sam_image = cv2.imread(image_path)
        sam_image = cv2.cvtColor(sam_image, cv2.COLOR_BGR2RGB)

        # YOLO preprocess
        im = cv2.resize(sam_image, (640, 640))
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float().to(device) / 255.0
        im = im.unsqueeze(0)

        # YOLO inference
        pred = yolo(im)
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, max_det=300)

        if len(pred[0]) == 0:
            continue

        det = pred[0]

        # scale boxes back
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], sam_image.shape).round()

        boxes = det[:, :4]
        scores = det[:, 4]
        labels = det[:, 5].int()
        #labels = torch.zeros_like(det[:, 5]).int() # for single class

        # ---------------- SAM ---------------- #
        predictor.set_image(sam_image)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes, sam_image.shape[:2]).to(device)
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        masks = masks.cpu().numpy()
        evaluator.update(image_id,boxes.cpu(),scores.cpu(),labels.cpu(),masks)            
        if (i + 1) % 50 == 0:
            used = time.time() - start
            print(f"{i}/{len(dataset)} processed, time={used:.1f}s")

        iou_matrix = compute_iou_matrix(masks, gt_masks)
        best_ious = iou_matrix.max(axis=1) if len(gt_masks) > 0 else np.zeros(len(masks))
        # ---------------- Visualization ---------------- #
        origin_path = os.path.join(origin_dir, file_name)
        cv2.imwrite(origin_path, cv2.cvtColor(sam_image, cv2.COLOR_RGB2BGR))
        
        yolo_path = os.path.join(yolo_dir, file_name)
        #visualize_yolo(sam_image, boxes, scores, yolo_path)
        visualize_yolo(sam_image,boxes, scores,labels.cpu().numpy(),cat_id_to_name,yolo_path)
        
        yolo_sam_path = os.path.join(yolo_sam_dir, file_name)
        #visualize_and_save(sam_image,boxes, scores,masks,yolo_sam_path)
        visualize_and_save(sam_image, boxes, scores, masks, labels.cpu().numpy(),cat_id_to_name, yolo_sam_path, best_ious)
        
        gt_path = os.path.join(gt_dir, file_name)
        visualize_gt(sam_image, gt_masks, gt_boxes, gt_labels, cat_id_to_name, gt_path)
        #evaluator.update_iou_dice(masks, gt_masks)
        evaluator.update_iou_dice(masks, gt_masks, labels.cpu().numpy())
        
    # eval bbox and segm metric, save json and txt
    evaluator.finalize(save_json=args.save_json)
    
# ---------------- Argparse ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--yolo_checkpoint", type=str, required=True)
    parser.add_argument("--sam_checkpoint", type=str, default=None)
    parser.add_argument("--sam_enhanced_checkpoint", type=str, help="path to sam-enhanced checkpoint file")
    parser.add_argument("--use_sam_enhanced", action="store_true", help="using sam-enhanced for prediction")
    
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf_thres", type=float, default=0.25)
    parser.add_argument("--iou_thres", type=float, default=0.45)

    parser.add_argument("--save_json", action="store_true")

    args = parser.parse_args()

    main(args)

