#!/usr/bin/env bash
set -euo pipefail

EUVP="${EUVP:?}"
UIEB="${UIEB:?}"
SUIM="${SUIM:?}"
USOD="${USOD:?}"
TEACH_A="${TEACH_A:?}"

count_imgs () { find "$1" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" -o -iname "*.webp" \) | wc -l; }

echo "========================="
echo "DATASET COUNTS (FULL)"
echo "========================="
echo "EUVP root: $EUVP"
echo "UIEB root: $UIEB"
echo "SUIM root: $SUIM"
echo "USOD root: $USOD"
echo "TeacherA:  $TEACH_A"
echo ""

echo "---- EUVP Paired ----"
for s in underwater_dark underwater_imagenet underwater_scenes; do
  echo "EUVP/Paired/$s/trainA      :" $(count_imgs "$EUVP/Paired/$s/trainA")
  echo "EUVP/Paired/$s/trainB      :" $(count_imgs "$EUVP/Paired/$s/trainB")
  echo "EUVP/Paired/$s/validation  :" $(count_imgs "$EUVP/Paired/$s/validation")
  echo ""
done

echo "---- EUVP Unpaired ----"
echo "EUVP/Unpaired/trainA       :" $(count_imgs "$EUVP/Unpaired/trainA")
echo "EUVP/Unpaired/trainB       :" $(count_imgs "$EUVP/Unpaired/trainB")
echo "EUVP/Unpaired/validation   :" $(count_imgs "$EUVP/Unpaired/validation")
echo ""

echo "---- EUVP test_samples ----"
echo "EUVP/test_samples/Inp      :" $(count_imgs "$EUVP/test_samples/Inp")
echo "EUVP/test_samples/GTr      :" $(count_imgs "$EUVP/test_samples/GTr")
echo ""

echo "---- UIEB ----"
echo "UIEB/raw-890               :" $(count_imgs "$UIEB/raw-890")
echo "UIEB/reference-890         :" $(count_imgs "$UIEB/reference-890")
echo "UIEB/challenging-60        :" $(count_imgs "$UIEB/challenging-60")
echo ""

echo "---- SUIM ----"
echo "SUIM/train_val/images      :" $(count_imgs "$SUIM/train_val/images")
echo "SUIM/TEST/images           :" $(count_imgs "$SUIM/TEST/images")
echo ""

echo "---- USOD ----"
echo "USOD/images                :" $(count_imgs "$USOD/images")
echo ""

echo "---- Teacher A ----"
echo "TeacherA total             :" $(count_imgs "$TEACH_A")
