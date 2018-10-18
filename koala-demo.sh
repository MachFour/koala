#! /bin/bash

KOALA_BIN=/home/max/uni/thesis/experiments/compiled/koala
DATASET_PATH=/home/max/uni/thesis/dataset
OUTDIR=/tmp

function welcome_message() {
	echo
	echo "#######################################################################"
	echo "## Automatic Extraction of Nutrition Information from Food Labelling ##"
	echo "##                                                                   ##"
	echo "##               Algorithm processing Walk-through                   ##"
	echo "#######################################################################"
	echo
	echo "This program will run the algorithm on a selected image, and show you the"
	echo "intermediate processing images, similar to the ones on the poster."
	echo
	echo "Please mouse over the image and press any key to step through the images."
	echo
	echo "At the end, the test result will be printed, which is a comparison between"
	echo "the detected table and the ground truth table (in text form)."
	echo
	echo
}

clear
welcome_message
echo
echo -n "Please choose an image by typing a number between 1 and 55: "

read
NUMBER="$REPLY"
clear

IMAGE_FILE="${DATASET_PATH}/${NUMBER}.jpg"
GROUND_TRUTH_FILE="${DATASET_PATH}/${NUMBER}.txt"
if [[ "${NUMBER}" -ge 1 && "${NUMBER}" -le 55 ]]; then
	echo
	#echo image file: $IMAGE_FILE
	#echo ground truth file: $GROUND_TRUTH_FILE
	#${KOALA_BIN}
	echo "Running algorithm, please wait..."
	${KOALA_BIN} "$IMAGE_FILE" -t "$GROUND_TRUTH_FILE" -o "${OUTDIR}/${NUMBER}" -v > /dev/null
	clear
	echo "Results:"
	cat "${OUTDIR}/${NUMBER}.test"
else
	echo
	echo -n "Sorry, I couldn't understand '$NUMBER'. "
	echo "The number needs to be between 1 and 55."
fi

echo
echo "Press any key to restart..."
read -n 1
