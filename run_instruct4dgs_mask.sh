#!/bin/bash

# ===================================================================
# Mask-Based Multi-Style 4D Editing Pipeline (Dual-Prompt)
#
# Usage:
#   bash run_instruct4dgs_mask.sh <dataset> <scene_name> <prompt_fg> <guidance_scale> <image_guidance_scale> [prompt_bg]
#
# Example:
#   bash run_instruct4dgs_mask.sh dynerf coffee_martini \
#       "Turn the person into a Van Gogh painting" 7.5 1.5 \
#       "Make the background look like a fauvism painting"
#
# If prompt_bg is not given, it defaults to "none" (background stays unchanged during SDS)
# ===================================================================

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <dataset> <scene_name> <prompt_fg> <guidance_scale> <image_guidance_scale> [prompt_bg]"
    exit 1
fi

DATASET="$1"
SCENE_NAME="$2"
PROMPT_FG="$3"
GUIDANCE_SCALE="$4"
IMAGE_GUIDANCE_SCALE="$5"
PROMPT_BG="${6:-none}"

PYTHON="/tmp2/martinlin/miniconda3/envs/Gaussians4D/bin/python"
DATA_DIR="./data/${DATASET}/time0_${SCENE_NAME}"
MODEL_PATH="./output/${DATASET}/${SCENE_NAME}"
PLY_BASE="${MODEL_PATH}/point_cloud/iteration_14000/point_cloud.ply"
EDIT_PLY="${MODEL_PATH}/point_cloud_3dedit/${PROMPT_FG}/iteration_1000/point_cloud.ply"
REFINE_PLY="${MODEL_PATH}/point_cloud_refine/${PROMPT_FG}/iteration_800/point_cloud.ply"

echo "=========================================="
echo "  Mask-Based Dual-Prompt 4D Editing"
echo "=========================================="
echo "  dataset:         ${DATASET}"
echo "  scene:           ${SCENE_NAME}"
echo "  prompt_fg:       \"${PROMPT_FG}\""
echo "  prompt_bg:       \"${PROMPT_BG}\""
echo "  guidance_scale:  ${GUIDANCE_SCALE}"
echo "  img_guidance:    ${IMAGE_GUIDANCE_SCALE}"
echo "=========================================="
echo ""

# ─── Step 1: Collect time0 images ──────────────────────────────
echo "[1/7] Collect time0 images..."
${PYTHON} time0_collect.py --dataset ${DATASET} --scene_name ${SCENE_NAME}
echo "✅ Completed."
echo ""

# ─── Step 2: SAM2 annotation (MANUAL) ─────────────────────────
echo "[2/7] SAM2 Annotation (MANUAL STEP)"
echo "  If you haven't annotated yet, run:"
echo "    ${PYTHON} scripts/annotate_server.py --data_dir ${DATA_DIR} --port 8765"
echo "  Then in browser: http://<server_ip>:8765"
echo "  After annotation, run mask generation:"
echo "    CUDA_VISIBLE_DEVICES=0 ${PYTHON} scripts/generate_masks_sam.py \\"
echo "      --data_dir ${DATA_DIR} \\"
echo "      --prompts_json ${DATA_DIR}/masks/custom_prompts.json"
echo ""
echo "  Press Enter to continue (assumes masks are already generated)..."
read -r
echo ""

# ─── Step 3: Generate hybrid images ──────────────────────────
echo "[3/7] Generate hybrid images (dual-style 2D composites)..."
${PYTHON} scripts/generate_hybrid_images.py \
    --data_dir "${DATA_DIR}" \
    --prompt_fg "${PROMPT_FG}" \
    --prompt_bg "${PROMPT_BG}" \
    --guidance_scale ${GUIDANCE_SCALE} \
    --image_guidance_scale ${IMAGE_GUIDANCE_SCALE}
echo "✅ Completed hybrid image generation."
echo ""

# ─── Step 4: 3D editing (pixel-level L1+SSIM) ────────────────
echo "[4/7] 3D Editing (pixel-level L1+SSIM fitting to hybrid images)..."
${PYTHON} edit_3d.py \
    --configs "./arguments/${DATASET}/${SCENE_NAME}.py" \
    --ply_path "${PLY_BASE}" \
    -s "./data/${DATASET}/${SCENE_NAME}" \
    --model_path "${MODEL_PATH}" \
    --dataset "${DATASET}" \
    --scene "${SCENE_NAME}" \
    --prompt "${PROMPT_FG}"
echo "✅ Completed 3D editing."
echo ""

# ─── Step 5: Inject 3D mask ──────────────────────────────────
echo "[5/7] Inject 3D mask from multi-camera SAM2 masks..."
${PYTHON} train_mask_full.py \
    --configs "./arguments/${DATASET}/${SCENE_NAME}.py" \
    -s "./data/${DATASET}/${SCENE_NAME}" \
    --model_path "${MODEL_PATH}" \
    --ply_path "${EDIT_PLY}"
echo "✅ Completed 3D mask injection."
echo ""

# ─── Step 6: Dual-prompt SDS Refinement ──────────────────────
echo "[6/7] Mask-based dual-prompt SDS refinement (temporal consistency)..."
${PYTHON} refine_sds.py \
    --configs "./arguments/${DATASET}/${SCENE_NAME}.py" \
    --ply_path "${EDIT_PLY}" \
    -s "./data/${DATASET}/${SCENE_NAME}" \
    --model_path "${MODEL_PATH}" \
    --prompt_fg "${PROMPT_FG}" \
    --prompt_bg "${PROMPT_BG}" \
    --guidance_scale ${GUIDANCE_SCALE} \
    --image_guidance_scale ${IMAGE_GUIDANCE_SCALE}
echo "✅ Completed SDS refinement."
echo ""

# ─── Step 7: Render final video ──────────────────────────────
echo "[7/7] Render edited 4D video..."
${PYTHON} render_edited4d.py \
    --configs "./arguments/${DATASET}/${SCENE_NAME}.py" \
    --model_path "${MODEL_PATH}" \
    --ply_path "${REFINE_PLY}"
echo "✅ Video saved."
echo ""

echo "🎉 All pipeline steps completed!"
echo "  Final video: ${MODEL_PATH}/edited_${SCENE_NAME}.mp4"
