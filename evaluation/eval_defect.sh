#!/bin/bash

DATA_DIR="./data/construction_defect_yolo"

YOLO_CKPT="pretrained_checkpoint/yolov5.pt"
SAM_CKPT="pretrained_checkpoint/sam.pth"

DEVICE="cuda:0"
CONF_THRES=0.25
IOU_THRES=0.45

ANNO_PATH="$DATA_DIR/annotations/val/normal.json"
IMAGE_DIR="$DATA_DIR/images/val"

if [ ! -f "$ANNO_PATH" ]; then
    echo "Annotation file not found: $ANNO_PATH"
    exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "Image directory not found: $IMAGE_DIR"
    exit 1
fi

python eval_on_defect_yolov5_sam.py \
    --yolo_checkpoint "$YOLO_CKPT" \
    --sam_enhanced_checkpoint "$SAM_CKPT" \
    --anno_path "$ANNO_PATH" \
    --image_dir "$IMAGE_DIR" \
    --device "$DEVICE" \
    --conf_thres $CONF_THRES \
    --iou_thres $IOU_THRES \
    --save_json \
    --use_sam_enhanced

echo "Done."
